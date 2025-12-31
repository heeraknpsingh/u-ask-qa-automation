"""
Configuration for LLM Evaluation.
"""
from enum import Enum
import os
from pathlib import Path
from typing import Optional
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class EvaluatorType(str, Enum):
    """Available evaluator backends."""
    RAGAS = "ragas"
    DEEPEVAL = "deepeval"
    HUGGINGFACE = "huggingface"

class EvaluatorConfig(BaseSettings):
    """Configuration for LLM evaluators."""
    model_config = SettingsConfigDict(
        case_sensitive=False,
        extra="ignore",
        # We load `.env` ourselves in a best-effort way (see `_load_dotenv_best_effort`)
        # so config initialization never fails if `.env` is unreadable in some environments.
        env_prefix="",
    )
    # API Keys (loaded from .env)
    openai_api_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("EVAL_OPENAI_API_KEY", "OPENAI_API_KEY"),
    )
    # Evaluator Selection
    evaluator_type: EvaluatorType = Field(
        default=EvaluatorType.RAGAS,
        validation_alias=AliasChoices("EVAL_EVALUATOR_TYPE", "EVALUATOR_TYPE"),
    )
    # Thresholds (0.0 to 1.0)
    # Prefer values from `.env` (EVAL_*). If not provided, `get_thresholds()` will
    # fall back to sane defaults so the evaluator still runs out-of-the-box.
    faithfulness_threshold: Optional[float] = Field(
        default=None,
        validation_alias=AliasChoices("EVAL_FAITHFULNESS_THRESHOLD", "FAITHFULNESS_THRESHOLD"),
        ge=0.0,
        le=1.0,
    )
    relevancy_threshold: Optional[float] = Field(
        default=None,
        validation_alias=AliasChoices("EVAL_RELEVANCY_THRESHOLD", "RELEVANCY_THRESHOLD"),
        ge=0.0,
        le=1.0,
    )
    context_precision_threshold: Optional[float] = Field(
        default=None,
        validation_alias=AliasChoices(
            "EVAL_CONTEXT_PRECISION_THRESHOLD",
            "CONTEXT_PRECISION_THRESHOLD",
        ),
        ge=0.0,
        le=1.0,
    )
    context_recall_threshold: Optional[float] = Field(
        default=None,
        validation_alias=AliasChoices("EVAL_CONTEXT_RECALL_THRESHOLD", "CONTEXT_RECALL_THRESHOLD"),
        ge=0.0,
        le=1.0,
    )
    answer_correctness_threshold: Optional[float] = Field(
        default=None,
        validation_alias=AliasChoices(
            "EVAL_ANSWER_CORRECTNESS_THRESHOLD",
            "ANSWER_CORRECTNESS_THRESHOLD",
        ),
        ge=0.0,
        le=1.0,
    )
    hallucination_threshold: Optional[float] = Field(
        default=None,
        validation_alias=AliasChoices("EVAL_HALLUCINATION_THRESHOLD", "HALLUCINATION_THRESHOLD"),
        ge=0.0,
        le=1.0,
    )
    toxicity_threshold: Optional[float] = Field(
        default=None,
        validation_alias=AliasChoices("EVAL_TOXICITY_THRESHOLD", "TOXICITY_THRESHOLD"),
        ge=0.0,
        le=1.0,
    )
    bias_threshold: Optional[float] = Field(
        default=None,
        validation_alias=AliasChoices("EVAL_BIAS_THRESHOLD", "BIAS_THRESHOLD"),
        ge=0.0,
        le=1.0,
    )
    similarity_threshold: Optional[float] = Field(
        default=None,
        validation_alias=AliasChoices("EVAL_SIMILARITY_THRESHOLD", "SIMILARITY_THRESHOLD"),
        ge=0.0,
        le=1.0,
    )
    nli_entailment_threshold: Optional[float] = Field(
        default=None,
        validation_alias=AliasChoices(
            "EVAL_NLI_ENTAILMENT_THRESHOLD",
            "NLI_ENTAILMENT_THRESHOLD",
        ),
        ge=0.0,
        le=1.0,
    )

    def get_thresholds(self) -> dict:
        """Return all thresholds as a dictionary."""
        defaults = {
            "faithfulness": 0.7,
            "relevancy": 0.7,
            "context_precision": 0.7,
            "context_recall": 0.7,
            "answer_correctness": 0.7,
            "hallucination": 0.5,
            "toxicity": 0.3,
            "bias": 0.3,
            "similarity": 0.7,
            "nli_entailment": 0.5,
        }
        return {
            "faithfulness": self.faithfulness_threshold
            if self.faithfulness_threshold is not None
            else defaults["faithfulness"],
            "relevancy": self.relevancy_threshold
            if self.relevancy_threshold is not None
            else defaults["relevancy"],
            "context_precision": self.context_precision_threshold
            if self.context_precision_threshold is not None
            else defaults["context_precision"],
            "context_recall": self.context_recall_threshold
            if self.context_recall_threshold is not None
            else defaults["context_recall"],
            "answer_correctness": self.answer_correctness_threshold
            if self.answer_correctness_threshold is not None
            else defaults["answer_correctness"],
            "hallucination": self.hallucination_threshold
            if self.hallucination_threshold is not None
            else defaults["hallucination"],
            "toxicity": self.toxicity_threshold
            if self.toxicity_threshold is not None
            else defaults["toxicity"],
            "bias": self.bias_threshold
            if self.bias_threshold is not None
            else defaults["bias"],
            "similarity": self.similarity_threshold
            if self.similarity_threshold is not None
            else defaults["similarity"],
            "nli_entailment": self.nli_entailment_threshold
            if self.nli_entailment_threshold is not None
            else defaults["nli_entailment"],
        }

def _load_dotenv_best_effort(path: str = ".env") -> None:
    """
    Best-effort `.env` loader.
    - If `.env` is readable, load key/value pairs into the process env
      (without overriding existing env vars)
    - If `.env` is missing/unreadable, do nothing (no crash)
    """
    try:
        raw = Path(path).read_text(encoding="utf-8")
    except OSError:
        return

    for line in raw.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if "=" not in s:
            continue
        key, value = s.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        # Strip optional surrounding quotes.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        os.environ.setdefault(key, value)

def get_evaluator_config() -> EvaluatorConfig:
    """Get the evaluator configuration (loads `.env` first when possible)."""
    # Always resolve `.env` relative to the repository root (not the current CWD).
    env_path = Path(__file__).resolve().parents[3] / ".env"
    _load_dotenv_best_effort(str(env_path))
    return EvaluatorConfig()
