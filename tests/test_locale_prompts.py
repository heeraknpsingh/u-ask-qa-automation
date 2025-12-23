"""
Language prompt smoke tests driven by the first two prompts in data/test-data.json.
"""

from __future__ import annotations
from pathlib import Path
import pytest
from u_ask_qa.pages import ChatbotPage
from u_ask_qa.validators.response_validator import ResponseValidator


def _load_first_two_prompts() -> tuple[dict, dict]:
    """
    Load the first two prompt entries from data/test-data.json.
    Requirement: "pick the prompt from test json file first 2".
    """
    repo_root = Path(__file__).resolve().parents[1]
    data_path = repo_root / "data" / "test-data.json"
    import json

    data = json.loads(data_path.read_text(encoding="utf-8"))
    prompts = data.get("prompts", [])
    assert len(prompts) >= 2, "test-data.json must contain at least 2 prompts"
    return prompts[0], prompts[1]


@pytest.mark.smoke
@pytest.mark.response
class TestLocalePrompts:
    async def test_chat_with_english_prompt_from_json(
        self,
        chatbot_page: ChatbotPage,
        response_validator: ResponseValidator,
    ):
        p0, _ = _load_first_two_prompts()
        assert p0.get("lang") == "en", (
            "First prompt in test-data.json is expected to be English"
        )
        await chatbot_page.set_locale_in_local_storage("en", reload_page=False)
        await chatbot_page.send_message(p0["q"])
        await chatbot_page.wait_for_response()
        responses = await chatbot_page.get_all_responses()
        assert responses, "No response received from bot"
        result = response_validator.validate_response(responses[-1], expected_lang="en")
        assert result.is_valid, f"Issues found: {', '.join(result.issues)}"

    async def test_chat_with_arabic_prompt_from_json(
        self,
        chatbot_page: ChatbotPage,
        response_validator: ResponseValidator,
    ):
        _, p1 = _load_first_two_prompts()
        assert p1.get("lang") == "ar", (
            "Second prompt in test-data.json is expected to be Arabic"
        )
        await chatbot_page.set_locale_in_local_storage("ar", reload_page=True)
        await chatbot_page.send_message(p1["q"])
        await chatbot_page.wait_for_response()
        responses = await chatbot_page.get_all_responses()
        assert responses, "No response received from bot"
        result = response_validator.validate_response(responses[-1], expected_lang="ar")
        assert result.is_valid, f"Issues found: {', '.join(result.issues)}"
