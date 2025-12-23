"""Tests for AI response quality and accuracy."""

import time
import pytest
from u_ask_qa.pages import ChatbotPage


class TestResponseQuality:
    """Validates that the AI provides helpful and relevant answers."""

    @pytest.mark.response
    @pytest.mark.smoke
    async def test_basic_query_response(
        self, chatbot_page: ChatbotPage, response_validator
    ):
        """Verify AI response for a standard visa query."""
        await chatbot_page.send_message("How do I apply for a UAE tourist visa?")
        await chatbot_page.wait_for_response()
        responses = await chatbot_page.get_all_responses()
        assert responses, "No response received from bot"
        # Validate the last response
        result = response_validator.validate_response(responses[-1], expected_lang="en")
        assert result.is_valid, f"Issues found: {', '.join(result.issues)}"

    @pytest.mark.response
    async def test_response_timing(self, chatbot_page: ChatbotPage, test_settings):
        """Verify the bot responds within the configured timeout."""
        start_time = time.time()
        await chatbot_page.send_message("Hello")
        await chatbot_page.wait_for_response()
        duration_ms = (time.time() - start_time) * 1000
        assert duration_ms < test_settings.response_timeout, (
            f"Response too slow: {duration_ms}ms"
        )
