"""
LLM Response Evaluation Module.

Provides evaluation capabilities using RAGAS, DeepEval, and HuggingFace
for hallucination detection, relevancy scoring, and response quality assessment.
"""

from .config import EvaluatorConfig, EvaluatorType, get_evaluator_config
from .llm_evaluator import (
    LLMEvaluator,
    EvaluationResult,
    EvaluationMetrics,
    MetricResult,
    MetricStatus,
)

__all__ = [
    "LLMEvaluator",
    "EvaluationResult",
    "EvaluationMetrics",
    "MetricResult",
    "MetricStatus",
    "EvaluatorType",
    "EvaluatorConfig",
    "get_evaluator_config",
]
