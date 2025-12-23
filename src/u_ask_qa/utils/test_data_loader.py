"""Utility for loading and filtering test prompts and security payloads from JSON."""

import json
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field


class TestPrompt(BaseModel):
    """Represents a standard user query and expected response type."""

    id: str
    q: str = Field(alias="q")  # Mapping 'q' from JSON to a clearer field
    lang: str = "en"
    tags: List[str] = Field(default_factory=list)
    expect: str = "response"


class SecurityPayload(BaseModel):
    """Represents a malicious payload used for security/vulnerability testing."""

    id: str
    p: str = Field(alias="p")  # Mapping 'p' from JSON
    t: str = Field(alias="t")  # Mapping 't' (type) from JSON


class TestDataLoader:
    """Loads and filters test data from the project data directory."""

    def __init__(self, data_dir: Path = Path("data")):
        self.data_dir = data_dir
        self._data = None

    def _load(self) -> dict:
        """Lazy-loads the JSON data file."""
        if self._data is None:
            filepath = self.data_dir / "test-data.json"
            with open(filepath, "r", encoding="utf-8") as f:
                self._data = json.load(f)
        return self._data

    def get_prompts(
        self, lang: Optional[str] = None, tags: Optional[List[str]] = None
    ) -> List[TestPrompt]:
        """Filters prompts by language and/or tags."""
        data = self._load()
        prompts = [TestPrompt(**item) for item in data.get("prompts", [])]

        if lang:
            prompts = [p for p in prompts if p.lang == lang]
        if tags:
            prompts = [p for p in prompts if any(t in p.tags for t in tags)]

        return prompts

    def get_security_payloads(
        self, p_type: Optional[str] = None
    ) -> List[SecurityPayload]:
        """Filters security payloads by type (e.g., 'xss', 'injection')."""
        data = self._load()
        payloads = [SecurityPayload(**item) for item in data.get("security", [])]

        if p_type:
            payloads = [p for p in payloads if p.t == p_type]

        return payloads

    def get_prompt_pairs(self):
        """Pairs English and Arabic prompts with the same base ID for cross-language testing."""
        prompts = self.get_prompts()
        en_map = {p.id.replace("_en", ""): p for p in prompts if p.id.endswith("_en")}
        ar_map = {p.id.replace("_ar", ""): p for p in prompts if p.id.endswith("_ar")}

        pairs = []
        for base_id, en_prompt in en_map.items():
            if base_id in ar_map:
                pairs.append((en_prompt, ar_map[base_id]))
        return pairs
