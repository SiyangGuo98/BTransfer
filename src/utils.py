import os
from typing import Any, Callable, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForMaskedLM, AutoConfig, EvalPrediction, Trainer, TrainingArguments

from .models import RobertaForPromptFinetuning
from .processors import compute_metrics_mapping


def load_models_from_cache(
    model_pt_old_name: str, 
    model_pt_new_name: str, 
    dataset_name: str, 
    cache_dir: str,
) -> Tuple[nn.Module, nn.Module, nn.Module]:
    """Load pre-trained and fine-tuned models from cache directory."""
    model_pt_old_path = os.path.join(cache_dir, "models/lm-bff", model_pt_old_name)
    config = AutoConfig.from_pretrained(model_pt_old_path)
    config.tie_word_embeddings = False
    model_pt_old = AutoModelForMaskedLM.from_pretrained(
        model_pt_old_path,
        config=config
    )
    
    model_ft_old_path = os.path.join(
        cache_dir, "models/lm-bff/roberta_ckpts",
        f"{dataset_name}-prompt-{model_pt_old_name}"
    )
    model_ft_old = RobertaForPromptFinetuning.from_pretrained(model_ft_old_path)

    model_pt_new_path = os.path.join(cache_dir, "models/lm-bff", model_pt_new_name)
    model_pt_new = AutoModelForMaskedLM.from_pretrained(model_pt_new_path)
    
    return model_pt_old, model_ft_old, model_pt_new


def build_compute_metrics_fn(
    task_name: str,
    test_dataset: Any
) -> Callable[[EvalPrediction], Dict]:
    """Build compute metrics function for evaluation."""
    def compute_metrics_fn(p: EvalPrediction) -> Dict:
        predictions = p.predictions
        num_logits = predictions.shape[-1]
        logits = predictions.reshape([test_dataset.num_sample, -1, num_logits])
        logits = logits.mean(axis=0)
        
        if num_logits == 1:
            preds = np.squeeze(logits)
        else:
            preds = np.argmax(logits, axis=1)

        label_ids = p.label_ids.reshape([test_dataset.num_sample, -1])
        label_ids_avg = label_ids.mean(axis=0)
        label_ids_avg = label_ids_avg.astype(p.label_ids.dtype)
        assert (label_ids_avg - label_ids[0]).mean() < 1e-2
        label_ids = label_ids[0]
        
        return compute_metrics_mapping[task_name](task_name, preds, label_ids)

    return compute_metrics_fn


def evaluate_model(
    model: nn.Module,
    test_dataset: Any,
    training_args: TrainingArguments,
    compute_metrics_fn: Callable[[EvalPrediction], Dict]
) -> Dict:
    """Evaluate model on test dataset."""
    model.label_word_list = torch.tensor(test_dataset.label_word_list).long()
    
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics_fn
    )
    
    metrics = trainer.evaluate(eval_dataset=test_dataset)
    return metrics