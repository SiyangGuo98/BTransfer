from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List
import yaml

DEFAULT_EXCLUDE_PATTERNS = ['lm_head', 'model.embed', 'pooler']


@dataclass
class BaseHparams:
    """Base hyperparameters for transfer methods."""
    model_pt_old_name: str
    model_pt_new_name: str
    fine_tuning_dataset_name: str
    transfer_method: str
    exclude_param_names: List[str] = field(
        default_factory=lambda: DEFAULT_EXCLUDE_PATTERNS.copy()
    )

    @classmethod
    def from_yaml(
        cls,
        yaml_path: str,
        **base_params: Any
    ) -> 'BaseHparams':
        """Load hyperparameters from YAML file."""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            yaml_dict = yaml.safe_load(f) or {}
        merged_dict = {**yaml_dict, **base_params}
        return cls(**merged_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert hyperparameters to dictionary."""
        return asdict(self)
    
    def update(self, params: Dict[str, Any]) -> None:
        """Update hyperparameters from dictionary."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)


class BaseTransfer:
    """Base class for transfer methods."""
    
    def __init__(self, hparams: BaseHparams) -> None:
        self.hparams = hparams
        self.exclude_param_names = hparams.exclude_param_names or []
    
    def should_exclude_param(self, param_name: str) -> bool:
        """Check if a parameter should be excluded from transfer."""
        return any(pattern in param_name for pattern in self.exclude_param_names)

    def get_transferred_model(
        self,
        model_pt_old: Any,
        model_ft_old: Any,
        model_pt_new: Any,
        **kwargs: Any
    ) -> Any:
        """Get transferred model. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement get_transferred_model().")
