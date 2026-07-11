"""Synthetic VQA smoke and Zoo-Bus-compatible construction pipelines."""

from .pipeline import run_reference_pipeline, run_smoke_pipeline
from .reference import ReferenceBuildConfig

__all__ = ["ReferenceBuildConfig", "run_reference_pipeline", "run_smoke_pipeline"]
__version__ = "0.2.0"
