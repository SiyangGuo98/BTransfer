import os
from typing import List, Optional, Tuple

import torch
import transformers
from datasets import Dataset, load_from_disk, load_dataset
from transformers import DataCollatorForLanguageModeling

from constants import CACHE_DIR


class WikiDataLoader:
    """Data loader for Wikipedia dataset used in Fisher computation."""
    
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer) -> None:
        self.tokenizer = tokenizer

    def load_dataset(
        self,
        sample_size: int = 5000,
        max_seq_length: int = 128,
    ) -> Dataset:
        """Load and preprocess Wikipedia dataset."""
        dataset_path = os.path.join(CACHE_DIR, "datasets/wikipedia")
        try:
            dataset = load_from_disk(dataset_path)['train']
        except Exception:
            dataset = load_dataset("wikipedia", "20220301.en", cache_dir=CACHE_DIR)
            dataset.save_to_disk(dataset_path)
            dataset = load_from_disk(dataset_path)['train']
        dataset = dataset.shuffle(seed=42).select(range(sample_size))
        dataset = dataset.map(
            lambda examples: self.tokenizer(
                text=examples["text"],
                max_length=max_seq_length,
                truncation=True
            ),
            batched=True,
            remove_columns=["title", "text"]
        )
        return dataset

class DeterministicDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    """MLM data collator with deterministic masking based on input content."""

    def __init__(
        self,
        *args,
        base_seed: int = 42,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.base_seed = base_seed

    @staticmethod
    def _stable_hash(ids: List[int]) -> int:
        """Compute FNV-1a hash from token IDs for deterministic seeding."""
        h = 2166136261  # FNV offset basis
        for x in ids:
            h ^= int(x) & 0xFFFFFFFF
            h = (h * 16777619) & 0xFFFFFFFF  # FNV prime
        return h

    def torch_mask_tokens(
        self,
        inputs: torch.Tensor,
        special_tokens_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply MLM masking with deterministic behavior based on input content."""
        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token. "
                "Pass `mlm=False` to train on causal language modeling instead."
            )

        labels = inputs.clone()
        device = inputs.device

        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(
                    val, already_has_special_tokens=True
                )
                for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(
                special_tokens_mask, dtype=torch.bool, device=device
            )
        else:
            special_tokens_mask = special_tokens_mask.to(device=device, dtype=torch.bool)

        probability_matrix = torch.full(
            labels.shape,
            self.mlm_probability,
            dtype=torch.float,
            device=device,
        )
        probability_matrix.masked_fill_(special_tokens_mask, 0.0)

        batch_size, seq_len = inputs.shape
        masked_indices = torch.zeros_like(inputs, dtype=torch.bool, device=device)


        for i in range(batch_size):
            ids_i = inputs[i]
            pm_i = probability_matrix[i]

            h = self._stable_hash(ids_i.tolist())
            seed = (self.base_seed + h) & 0xFFFFFFFF

            g = torch.Generator(device=device)
            g.manual_seed(seed)

            mi = torch.bernoulli(pm_i, generator=g).to(torch.bool)
            masked_indices[i] = mi

            labels[i][~mi] = -100

            if not mi.any():
                continue

            # 80% -> [MASK]
            replace_prob = torch.full_like(ids_i, 0.8, dtype=torch.float)
            indices_replaced = (
                torch.bernoulli(replace_prob, generator=g).to(torch.bool) & mi
            )
            ids_i[indices_replaced] = self.tokenizer.mask_token_id

            # 10% -> random token
            rand_prob = torch.full_like(ids_i, 0.5, dtype=torch.float)
            indices_random = (
                torch.bernoulli(rand_prob, generator=g).to(torch.bool)
                & mi
                & ~indices_replaced
            )
            if indices_random.any():
                random_words = torch.randint(
                    low=0,
                    high=len(self.tokenizer),
                    size=ids_i.shape,
                    dtype=torch.long,
                    device=device,
                    generator=g,
                )
                ids_i[indices_random] = random_words[indices_random]


        return inputs, labels