"""RIGOR: attention-calibrated leakage detection and safe generation."""

from .generation import (
    AttentionLeakageDetector,
    DecodeSettings,
    LeakageSafeGenerator,
    RepairSettings,
)

__all__ = [
    "AttentionLeakageDetector",
    "DecodeSettings",
    "LeakageSafeGenerator",
    "RepairSettings",
]
