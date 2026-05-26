import copy
import math
from dataclasses import dataclass
from itertools import islice
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments
from tqdm import tqdm

from ..base_transfer import BaseHparams, BaseTransfer
from .utils import WikiDataLoader, DeterministicDataCollatorForLanguageModeling


@dataclass
class TransferHparams(BaseHparams):
    """Hyperparameters for BTransfer method."""
    fisher_sample_size: int = 5000
    fisher_batch_size: int = 16


class Transfer(BaseTransfer):
    """BTransfer transfer method using the reusable task-update factor. """
    
    def __init__(self, hparams: TransferHparams) -> None:
        super().__init__(hparams)
        self.fisher_sample_size = hparams.fisher_sample_size
        self.fisher_batch_size = hparams.fisher_batch_size
        self.fisher_epsilon = 1e-8

    def get_trainer_for_fisher(
        self, 
        model: nn.Module,
        dataset: Any, 
        tokenizer: Any,
        task_type: str = "mlm"
    ) -> Trainer:
        """Create trainer for Fisher information computation."""
        training_args = TrainingArguments(
            output_dir="output/",
            per_device_train_batch_size=self.fisher_batch_size,
            dataloader_num_workers=4,
        )

        if task_type == "prompt":
            model.label_word_list = torch.tensor(dataset.label_word_list).long().cuda()
            return Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
            )
        elif task_type == "mlm":
            data_collator = DeterministicDataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=True,
                mlm_probability=0.15,
                base_seed=42
            )
            return Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                data_collator=data_collator,
            )
        else:
            raise ValueError(f"Unsupported task type: {task_type}")


    def compute_diagonal_fisher(
        self,
        trainer: Trainer,
        use_labels_mask: bool = True,                                                     
    ) -> Dict[str, torch.Tensor]:
        """Compute diagonal Fisher information matrix."""
        model = trainer.model
        model.eval()

        def _count_valid_tokens(batch: Dict[str, Any]) -> int:
            if use_labels_mask and ("labels" in batch) and (batch["labels"] is not None):
                pos_mask = (batch["labels"] != -100)
            elif ("attention_mask" in batch) and (batch["attention_mask"] is not None):
                pos_mask = (batch["attention_mask"] > 0)
            elif ("input_ids" in batch) and (batch["input_ids"] is not None):
                pos_mask = torch.ones_like(batch["input_ids"], dtype=torch.bool)
            else:
                raise ValueError("Unable to determine valid tokens for Fisher computation.")
            return int(pos_mask.reshape(-1).sum().item())

        device = "cuda" if torch.cuda.is_available() else "cpu"
        fisher: Dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            if (not p.requires_grad) or self.should_exclude_param(name):
                continue
            fisher[name] = torch.zeros_like(p, device=device, dtype=torch.float32)


        train_loader = trainer.get_train_dataloader()
        num_batches = math.ceil(self.fisher_sample_size / self.fisher_batch_size)
        limited_train_loader = islice(train_loader, num_batches)
        total_samples = 0
        total_valid_tokens = 0

        with torch.enable_grad():
            for batch in tqdm(limited_train_loader, total=num_batches, desc="Computing Fisher"):

                batch = trainer._prepare_inputs(batch)
                model.zero_grad(set_to_none=True)

                if "input_ids" not in batch or batch["input_ids"] is None:
                    raise ValueError("batch must contain input_ids to count samples.")
                batch_size = int(batch["input_ids"].shape[0])
                total_samples += batch_size
                n_valid = _count_valid_tokens(batch)
                total_valid_tokens += n_valid

                outputs = model(**batch)
                loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
                if n_valid > 0:
                    loss = loss * n_valid
                loss.backward()

                for name, p in model.named_parameters():
                    if name not in fisher:
                        continue
                    if p.grad is None:
                        raise ValueError(
                            f"Missing gradient for parameter '{name}' during Fisher computation"
                        )
                    fisher[name] += (p.grad.detach().float() ** 2)

        denom = float(total_valid_tokens)
        if denom <= 0:
            raise RuntimeError("No valid samples/tokens were processed; cannot normalize Fisher.")
        for name in fisher:
            fisher[name] = (fisher[name] / denom).cpu()

        print(
            f"Diagonal Fisher computed on {total_samples} samples; "
            f"{total_valid_tokens} valid tokens; normalized by token."
        )
        return fisher

    def _compute_fisher(
        self,
        model: nn.Module,
        dataset: Any,
        tokenizer: Any,
        task_type: str,
        description: str
    ) -> Dict[str, torch.Tensor]:
        """Compute Fisher information for a model on a dataset."""
        print("\n" + "="*60)
        print(description)
        print("="*60)
        trainer = self.get_trainer_for_fisher(
            model, dataset, tokenizer, task_type=task_type
        )
        return self.compute_diagonal_fisher(trainer)

    def get_fisher_matrices(
        self, 
        model_pt_old: nn.Module, 
        model_ft_old: nn.Module, 
        model_pt_new: nn.Module,
        **kwargs: Any
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Compute Fisher matrices for all three models."""
        if kwargs.get("benchmark") == "prompt-based classification":
            task_type_list = ["mlm", "prompt"]
        else:
            raise ValueError(
                f"Unsupported benchmark '{kwargs.get('benchmark')}' for BTransfer."
            )

        tokenizer = kwargs.get("tokenizer")
        wiki_data_loader = WikiDataLoader(tokenizer=tokenizer)
        wiki_dataset = wiki_data_loader.load_dataset(sample_size=self.fisher_sample_size, max_seq_length=128)

        # Step 1: Compute Fisher for model_pt_old on Wikipedia
        fisher_pt_old = self._compute_fisher(
            model_pt_old, wiki_dataset, tokenizer, task_type=task_type_list[0],
            description="Step 4.1: Computing Fisher for model_pt_old on Wikipedia"
        )
        
        # Step 2: Compute Fisher for model_ft_old on downstream task
        fine_tuning_dataset = kwargs.get("fine_tuning_dataset")
        
        fisher_ft_old_c = self._compute_fisher(
            model_ft_old, fine_tuning_dataset, tokenizer, task_type=task_type_list[1],
            description="Step 4.2: Computing Fisher for model_ft_old on downstream task"
        )
        
        # Step 3: Compute Fisher for model_pt_new on Wikipedia
        fisher_pt_new = self._compute_fisher(
            model_pt_new, wiki_dataset, tokenizer, task_type=task_type_list[0],
            description="Step 4.3: Computing Fisher for model_pt_new on Wikipedia"
        )

        return fisher_pt_old, fisher_ft_old_c, fisher_pt_new

    def get_transferred_model(
        self, 
        model_pt_old: nn.Module, 
        model_ft_old: nn.Module, 
        model_pt_new: nn.Module,
        **kwargs: Any
    ) -> nn.Module:
        """Get transferred model using BTransfer."""
        fisher_pt_old, fisher_ft_old_c, fisher_pt_new = self.get_fisher_matrices(
            model_pt_old,
            model_ft_old,
            model_pt_new,
            **kwargs
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"

        transferred_model = copy.deepcopy(model_ft_old)
        
        model_pt_old_state = model_pt_old.to(device).state_dict()
        model_ft_old_state = model_ft_old.to(device).state_dict()
        model_pt_new_state = model_pt_new.to(device).state_dict()
        transferred_state = transferred_model.to(device).state_dict()
        
        for param_name in tqdm(transferred_state.keys(), desc="BTransfer"):
            if not self.should_exclude_param(param_name):
                orig_dtype = transferred_state[param_name].dtype

                f_pt_old = torch.maximum(fisher_pt_old[param_name], torch.tensor(self.fisher_epsilon)).to(device=device, dtype=torch.float32)
                f_pt_new = torch.maximum(fisher_pt_new[param_name], torch.tensor(self.fisher_epsilon)).to(device=device, dtype=torch.float32)
                f_ft_old = torch.maximum(fisher_ft_old_c[param_name], torch.tensor(self.fisher_epsilon)).to(device=device, dtype=torch.float32)

                theta_pt_old = model_pt_old_state[param_name].to(dtype=torch.float32)
                theta_pt_new = model_pt_new_state[param_name].to(dtype=torch.float32)
                theta_ft_old = model_ft_old_state[param_name].to(dtype=torch.float32)

                numerator = (f_ft_old + f_pt_old) * theta_ft_old - f_pt_old * theta_pt_old + f_pt_new * theta_pt_new
                denominator = (f_ft_old + f_pt_old) - f_pt_old + f_pt_new
                transferred_state[param_name] = (numerator / denominator).to(dtype=orig_dtype)
            else:
                continue
            
        transferred_model.load_state_dict(transferred_state)

        print("\nBTransfer completed!")
        return transferred_model
