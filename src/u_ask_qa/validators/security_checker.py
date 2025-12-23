"""
Security validation for chatbot inputs and outputs.
Detects potential XSS, injection, and sanitization failures.
"""

import logging
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class SecurityCheckResult:
    """Outcome of a security test execution."""

    is_secure: bool
    category: str
    input_payload: str
    issues: List[str] = field(default_factory=list)


class SecurityChecker:
    """
    Analyzes bot responses to ensure they don't reflect or execute
    malicious payloads provided in the input.
    """

    def check_response_security(
        self, payload: str, response: str, category: str
    ) -> SecurityCheckResult:
        """
        Evaluates if a specific security payload was handled safely.
        Returns 'is_secure=False' if the payload is rendered verbatim in the output.
        """
        issues = []
        is_secure = True

        # Check for direct reflection (Classic XSS indicator)
        if payload in response:
            is_secure = False
            issues.append(f"Payload reflected in response: {category}")
            logger.warning(
                "Security issue detected (category=%s): payload reflected", category
            )

        # Additional logic for specific injection types could be added here
        return SecurityCheckResult(
            is_secure=is_secure,
            category=category,
            input_payload=payload,
            issues=issues,
        )
