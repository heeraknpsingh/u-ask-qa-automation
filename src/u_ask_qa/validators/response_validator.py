"""
Validation utilities for AI-generated responses.
Handles quality checks, relevance scoring, and basic hallucination detection.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

try:
    from langdetect import detect as _detect  # type: ignore[import-not-found]
    from langdetect.lang_detect_exception import (  # type: ignore[import-not-found]
        LangDetectException,
    )
except ImportError:  # pragma: no cover
    _detect = None

    class LangDetectException(Exception):
        """Fallback exception used when langdetect is not installed."""

        __slots__ = ()


logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Encapsulates the outcome of a response validation check."""

    is_valid: bool
    score: float  # 0.0 to 1.0
    issues: List[str] = field(default_factory=list)


class ResponseValidator:
    """
    Validates AI responses for quality, relevance, and presence of
    typical AI refusal or hallucination patterns.
    """

    # Indicators that the AI failed to provide a real answer
    REFUSAL_PATTERNS = [
        r"sorry.*try again",
        r"i don'?t understand",
        r"unable to process",
        r"error occurred",
        r"عذرًا",
        r"لم أفهم",
    ]

    # Indicators of standard AI constraints or "as an AI" logic
    AI_DISCLAIMERS = [
        r"as an ai",
        r"i don't have access",
        r"my training data",
        r"knowledge cutoff",
        r"i'm not able to",
    ]

    def validate_language(self, text: str, expected_lang: str) -> bool:
        """Checks if the response matches the expected ISO language code."""
        if _detect is None:
            logger.warning("langdetect is not available; skipping language detection")
            return True
        try:
            detected = _detect(text)
            is_match = detected == expected_lang
            if not is_match:
                logger.info(
                    "Language mismatch detected (expected=%s, detected=%s)",
                    expected_lang,
                    detected,
                )
            return is_match
        except LangDetectException:
            logger.warning("Language detection failed", exc_info=True)
            return False

    def check_for_refusal(self, text: str) -> List[str]:
        """Returns a list of detected refusal or error patterns."""
        issues = []
        for pattern in self.REFUSAL_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                issues.append(f"Refusal pattern detected: {pattern}")
        if issues:
            logger.info("Refusal patterns detected: %d", len(issues))
        return issues

    def check_hallucination_risk(self, text: str) -> List[str]:
        """Identifies patterns suggesting the AI is hallucinating or stating constraints."""
        issues = []
        # Check for AI disclaimers
        for pattern in self.AI_DISCLAIMERS:
            if re.search(pattern, text, re.IGNORECASE):
                issues.append(f"AI disclaimer/constraint found: {pattern}")
        # Check for absolute claims which often indicate low-quality reasoning
        absolute_claims = [r"100%", r"guaranteed", r"always", r"never", r"definitely"]
        for pattern in absolute_claims:
            if re.search(pattern, text, re.IGNORECASE):
                issues.append(f"Absolute claim detected: {pattern}")
        if issues:
            logger.info("Hallucination/disclaimer indicators detected: %d", len(issues))
        return issues

    def validate_response(
        self, text: str, expected_lang: Optional[str] = None
    ) -> ValidationResult:
        """
        Runs a full suite of validations on a given response string.
        """
        if not text or len(text.strip()) < 5:
            length = 0 if not text else len(text.strip())
            logger.info("Response is empty/too short (len=%s)", length)
            return ValidationResult(
                is_valid=False,
                score=0.0,
                issues=["Response is empty or too short"],
            )
        issues = []
        # 1. Language Check
        if expected_lang and not self.validate_language(text, expected_lang):
            issues.append(f"Language mismatch: Expected {expected_lang}")
        # 2. Refusal Check
        issues.extend(self.check_for_refusal(text))
        # 3. Hallucination/Disclaimer Check
        issues.extend(self.check_hallucination_risk(text))
        # Calculate a basic score (1.0 minus penalties for issues)
        score = max(0.0, 1.0 - (len(issues) * 0.2))
        hard_issues = [
            issue
            for issue in issues
            if not issue.startswith("Absolute claim detected:")
        ]
        logger.info(
            "Response validation complete (is_valid=%s, score=%.2f, issues=%d)",
            len(hard_issues) == 0,
            score,
            len(issues),
        )
        return ValidationResult(
            is_valid=len(hard_issues) == 0,
            score=score,
            issues=issues,
        )
