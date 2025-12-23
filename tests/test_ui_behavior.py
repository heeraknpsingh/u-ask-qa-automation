"""UI behavior and widget state tests."""

import pytest
from u_ask_qa.pages import ChatbotPage


class TestWidgetBehavior:
    """Tests for UI element states and basic navigation."""

    @pytest.mark.ui
    @pytest.mark.smoke
    async def test_widget_initial_load(self, chatbot_page: ChatbotPage):
        """Verify chat widget is visible on page load."""
        assert await chatbot_page.is_ready(), "Chat widget failed to initialize"

    @pytest.mark.ui
    async def test_error_handling_on_gibberish(self, chatbot_page: ChatbotPage):
        """Verify the bot handles nonsensical input without UI crash."""
        await chatbot_page.navigate()
        await chatbot_page.send_message("!@#$%^&*")
        await chatbot_page.wait_for_response()
        assert not await chatbot_page.has_error(), (
            "UI error message displayed for gibberish input"
        )
