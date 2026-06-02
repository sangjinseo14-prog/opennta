"""Batch tab — runs analysis across many spot files and emits per-group reports."""

from .output import TabBatchAnalysisOutput
from .tab import TabBatchAnalysis

__all__ = [
    "TabBatchAnalysis",
    "TabBatchAnalysisOutput",
]
