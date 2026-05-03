"""External benchmark helpers for public benchmark slices."""

from .external_runner import run_external_benchmark
from .grader import grade_predictions

__all__ = ["run_external_benchmark", "grade_predictions"]
