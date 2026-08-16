"""Dataset loaders."""

from pumpwatch.datasets.espset import (
    ESPSET_CITATION,
    ESPSET_LICENCE,
    espset_available,
    espset_order_features,
    load_espset,
)
from pumpwatch.datasets.ownrig import OwnRigRecord, OwnRigSessionMeta, load_ownrig, save_session
from pumpwatch.datasets.twente import (
    TWENTE_CITATION,
    collapse_labels,
    load_twente,
    write_demo_twente_cache,
)

__all__ = [
    "TWENTE_CITATION",
    "collapse_labels",
    "load_twente",
    "write_demo_twente_cache",
    "ESPSET_CITATION",
    "ESPSET_LICENCE",
    "espset_available",
    "espset_order_features",
    "load_espset",
    "OwnRigRecord",
    "OwnRigSessionMeta",
    "load_ownrig",
    "save_session",
]
