import copy
from dataclasses import dataclass
from typing import Any, Dict

import torch
import torch.nn as nn

from ..base_transfer import BaseHparams, BaseTransfer
from .permutation_spec import PermutationSpec, RobertaPermutationSpecBuilder
from .weights_matcher import LayerIterationOrder, WeightMatcher
from .utils import apply_permutation_to_statedict


@dataclass
class TransferHparams(BaseHparams):
    """Hyperparameters for TransFusion method."""
    lambda_: float = 1.0


class Transfer(BaseTransfer):
    """TransFusion transfer method using permutation-based weight matching."""
    
    def __init__(self, hparams: TransferHparams) -> None:
        super().__init__(hparams)
        self.lambda_ = hparams.lambda_

    def _get_model_type(
        self,
        state_dict: Dict[str, torch.Tensor]
    ) -> str:
        """Detect model type from state dict keys."""
        if any(key.startswith("roberta.encoder.layer.") for key in state_dict.keys()):
            return "roberta"
        raise ValueError("Unsupported model type for TransFusion.")

    def _get_num_heads(self, model: nn.Module) -> int:
        """Get number of attention heads from model config."""
        num_heads = getattr(model.config, "num_attention_heads", None)
        if num_heads is None:
            raise ValueError("Model config does not define num_attention_heads.")
        return int(num_heads)

    def _build_permutation_spec(
        self,
        model_type: str,
        depth: int
    ) -> PermutationSpec:
        """Build permutation specification for the model."""
        if model_type == "roberta":
            return RobertaPermutationSpecBuilder(depth=depth).create_permutation_spec()
        raise ValueError(f"Unsupported model type: {model_type}")

    def get_transferred_model(
        self,
        model_pt_old: nn.Module,
        model_ft_old: nn.Module,
        model_pt_new: nn.Module,
        **kwargs: Any
    ) -> nn.Module:
        """Get transferred model using permutation-based weight matching."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        transferred_model = copy.deepcopy(model_ft_old)

        model_pt_old_state = model_pt_old.to(device).state_dict()
        model_ft_old_state = model_ft_old.to(device).state_dict()
        model_pt_new_state = model_pt_new.to(device).state_dict()
        transferred_state = transferred_model.to(device).state_dict()

        model_type = self._get_model_type(model_pt_old_state)
        if model_type == "roberta":
            depth = len(model_pt_old.roberta.encoder.layer)
        else:
            depth = len(model_pt_old.model.layers)

        permutation_spec = self._build_permutation_spec(model_type, depth)
        num_heads = self._get_num_heads(model_pt_old)


        weight_matcher = WeightMatcher(
            ps=permutation_spec,
            max_iter=100,
            fixed=model_pt_new_state,
            permutee=model_pt_old_state,
            num_heads=num_heads,
            intra_head=True,
            layer_iteration_order=LayerIterationOrder.FORWARD,
            normalize_weights=True,
        )

        permutation, heads_permutation = weight_matcher.run()

        task_vector = {
            name: model_ft_old_state[name] - model_pt_old_state[name]
            for name in model_ft_old_state.keys()
            if not self.should_exclude_param(name)
        }

        permuted_task_vector = apply_permutation_to_statedict(
            permutation_spec,
            permutation,
            task_vector,
            heads_permutation=heads_permutation,
            num_heads=num_heads,
            skip_params=True,
        )


        for param_name in transferred_state.keys():
            if not self.should_exclude_param(param_name):
                orig_dtype = transferred_state[param_name].dtype

                pt_new_param = model_pt_new_state[param_name].to(dtype=torch.float32)
                permuted_task_vector_param = permuted_task_vector[param_name].to(
                    device=device, dtype=torch.float32
                )
                transferred_state[param_name] = (
                    pt_new_param + self.lambda_ * permuted_task_vector_param
                ).to(dtype=orig_dtype)
            else:
                continue

        transferred_model.load_state_dict(transferred_state)
        return transferred_model