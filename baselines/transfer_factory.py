from typing import Tuple, Type

from .base_transfer import BaseHparams, BaseTransfer

from .BTransfer.BTransfer import Transfer as BTransferClass, TransferHparams as BTransferHparams
from .paramD.paramD import Transfer as ParamDClass, TransferHparams as ParamDHparams
from .TransFusion.TransFusion import Transfer as TransFusionClass, TransferHparams as TransFusionHparams

_TRANSFER_REGISTRY = {
    'BTransfer': (BTransferClass, BTransferHparams),
    'paramD': (ParamDClass, ParamDHparams),
    'TransFusion': (TransFusionClass, TransFusionHparams),
}


def get_transfer_method(
    method_name: str
) -> Tuple[Type[BaseTransfer], Type[BaseHparams]]:
    """Get transfer class and hyperparameter class by method name."""
    if method_name not in _TRANSFER_REGISTRY:
        available = ', '.join(_TRANSFER_REGISTRY.keys())
        raise ValueError(
            f"Unknown transfer method '{method_name}'. "
            f"Available methods: {available}"
        )
    return _TRANSFER_REGISTRY[method_name]


def create_transfer_from_hparams(
    method_name: str,
    hparams: BaseHparams
) -> BaseTransfer:
    """Create transfer instance from hyperparameters."""
    transfer_cls, _ = get_transfer_method(method_name)
    return transfer_cls(hparams)

