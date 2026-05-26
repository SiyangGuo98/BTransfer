import random
import argparse
import numpy as np
import torch
from typing import Dict

from baselines.transfer_factory import get_transfer_method
from baselines.base_transfer import BaseHparams


def set_random_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_and_merge_hparams(args: argparse.Namespace) -> BaseHparams:
    """Load hyperparameters from YAML and merge with command-line arguments."""
    _, hparams_cls = get_transfer_method(args.transfer_method)
    
    base_params = {
        "model_pt_old_name": args.model_pt_old,
        "model_pt_new_name": args.model_pt_new,
        "fine_tuning_dataset_name": args.fine_tuning_dataset,
        "transfer_method": args.transfer_method
    }
    
    hparams_path = f"./hparams/{args.transfer_method}.yaml"
    hparams = hparams_cls.from_yaml(hparams_path, **base_params)
    
    hparam_fields = set(hparams.to_dict().keys()) - set(base_params.keys())
    for key in hparam_fields:
        value = getattr(args, key, None)
        if value is not None:
            setattr(hparams, key, value)
    
    return hparams


BENCHMARK_METRICS = {
    "prompt-based classification": {
        "sst-2": "acc",
        "mnli": "mnli/acc",
        "qnli": "acc",
        "rte": "acc",
        "subj": "acc",
        "mpqa": "acc",
        "mr": "acc",
    },
}

def standardize_metrics(
    metrics: Dict,
    dataset_name: str,
    benchmark: str
) -> Dict:
    """Standardize metrics output for consistent logging."""
    result = {}
    
    dataset_metrics = BENCHMARK_METRICS.get(benchmark, {})
    primary_name = dataset_metrics.get(dataset_name.lower())
    
    if primary_name is None:
        raise ValueError(
            f"Dataset {dataset_name} not found in benchmark {benchmark} metrics mapping."
        )
    result["primary_metric"] = metrics.get(f"eval_{primary_name}")
    result["primary_metric_name"] = primary_name
    
    return result