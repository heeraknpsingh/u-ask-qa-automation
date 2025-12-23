"""Security and injection testing for the chatbot."""
# pylint: disable=import-error, too-few-public-methods

import logging
import pytest  # type: ignore[import-not-found]
from u_ask_qa.pages import ChatbotPage  # type: ignore[import-not-found]

logger = logging.getLogger(__name__)


class TestSecurityPrevention:
    """Tests for XSS and Injection sanitization."""

    @pytest.mark.security
    async def test_payload_sanitization(
        self,
        chatbot_page: ChatbotPage,
        test_data,
        security_checker,
    ):
        """Tests various security payloads (XSS, Injection) for sanitization."""
        # Test a subset of security payloads
        payloads = test_data.get_security_payloads()[:5]
        for idx, p in enumerate(payloads, start=1):
            # Capture current count so we can reliably pick the response generated
            # for this payload (avoids mixing with earlier messages).
            before = len(await chatbot_page.get_all_responses())
            logger.info(
                "Security payload %d/%d | category=%s | before_responses=%d",
                idx,
                len(payloads),
                p.t,
                before,
            )
            await chatbot_page.send_message(p.p)
            await chatbot_page.wait_for_response()
            responses = await chatbot_page.get_all_responses()
            new_responses = responses[before:]
            last_reply = (
                new_responses[-1]
                if new_responses
                else (responses[-1] if responses else "")
            )
            if not new_responses:
                logger.warning(
                    "No new response captured for payload %d/%d (category=%s); "
                    "falling back to last known response",
                    idx,
                    len(payloads),
                    p.t,
                )
            result = security_checker.check_response_security(p.p, last_reply, p.t)
            logger.info(
                "Security check result | category=%s | is_secure=%s | issues=%d",
                p.t,
                result.is_secure,
                len(result.issues),
            )
            assert result.is_secure, (
                f"Security vulnerability ({p.t}) detected for payload: {p.p}"
            )
