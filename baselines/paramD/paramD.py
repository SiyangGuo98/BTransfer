import copy
from dataclasses import dataclass
from typing import Any, Dict

import torch
import torch.nn as nn

from ..base_transfer import BaseHparams, BaseTransfer


@dataclass
class TransferHparams(BaseHparams):
    """Hyperparameters for paramD transfer method."""
    lambda_: float = 1.0


class Transfer(BaseTransfer):
    """paramD transfer method using parameter difference."""
    
    def __init__(self, hparams: TransferHparams) -> None:
        super().__init__(hparams)
        self.lambda_ = hparams.lambda_
    
    def get_transferred_model(
        self,
        model_pt_old: nn.Module,
        model_ft_old: nn.Module,
        model_pt_new: nn.Module,
        **kwargs: Any
    ) -> nn.Module:
        """Transfer fine-tuning by applying paramD."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        transferred_model = copy.deepcopy(model_ft_old)
        
        model_pt_old_state = model_pt_old.to(device).state_dict()
        model_ft_old_state = model_ft_old.to(device).state_dict()
        model_pt_new_state = model_pt_new.to(device).state_dict()
        transferred_state = transferred_model.to(device).state_dict()

        for param_name in transferred_state.keys():
            if not self.should_exclude_param(param_name):
                orig_dtype = transferred_state[param_name].dtype

                pt_old_param = model_pt_old_state.get(param_name).to(dtype=torch.float32)
                ft_old_param = model_ft_old_state.get(param_name).to(dtype=torch.float32)
                pt_new_param = model_pt_new_state.get(param_name).to(dtype=torch.float32)

                
                delta = ft_old_param - pt_old_param
                transferred_state[param_name] = (
                    pt_new_param + self.lambda_ * delta
                ).to(dtype=orig_dtype)
            else:
                continue

        transferred_model.load_state_dict(transferred_state)
        return transferred_model


