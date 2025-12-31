"""Utility for loading and filtering test prompts and security payloads from JSON."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class TestPrompt(BaseModel):
    """Represents a standard user query and expected response type."""
    id: str
    q: str = Field(alias="q")
    lang: str = "en"
    tags: List[str] = Field(default_factory=list)
    expect: str = "response"

class SecurityPayload(BaseModel):
    """Represents a malicious payload used for security/vulnerability testing."""
    id: str
    p: str = Field(alias="p")
    t: str = Field(alias="t")

class EvaluationCase(BaseModel):
    """Represents a question with optional context and a reference answer for evaluation."""
    id: str
    question: str
    ground_truth: str
    context: List[str] = Field(default_factory=list)
    lang: str = "en"
    tags: List[str] = Field(default_factory=list)

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

    def get_prompts_as_dict(
        self, lang: Optional[str] = None, tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Get prompts as raw dictionaries."""
        data = self._load()
        prompts = data.get("prompts", [])
        if lang:
            prompts = [p for p in prompts if p.get("lang") == lang]
        if tags:
            prompts = [p for p in prompts if any(t in p.get("tags", []) for t in tags)]
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

    def get_security(self) -> List[Dict[str, Any]]:
        """Get all security payloads as raw dictionaries."""
        data = self._load()
        return data.get("security", [])

    def get_evaluation_cases(
        self,
        lang: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[EvaluationCase]:
        """Get curated evaluation cases (question + ground truth + optional context)."""
        data = self._load()
        cases = [EvaluationCase(**item) for item in data.get("evaluation", [])]
        if lang:
            cases = [c for c in cases if c.lang == lang]
        if tags:
            cases = [c for c in cases if any(t in c.tags for t in tags)]
        return cases

    def get_evaluation_cases_as_dict(
        self,
        lang: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Get evaluation cases as raw dictionaries."""
        data = self._load()
        cases = data.get("evaluation", [])
        if lang:
            cases = [c for c in cases if c.get("lang") == lang]
        if tags:
            cases = [c for c in cases if any(t in c.get("tags", []) for t in tags)]
        return cases

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

    # Convenience methods
    def get_smoke_prompts(self) -> List[Dict[str, Any]]:
        """Get prompts tagged with 'smoke'."""
        return self.get_prompts_as_dict(tags=["smoke"])

    def get_english_prompts(self) -> List[Dict[str, Any]]:
        """Get English language prompts."""
        return self.get_prompts_as_dict(lang="en")

    def get_arabic_prompts(self) -> List[Dict[str, Any]]:
        """Get Arabic language prompts."""
        return self.get_prompts_as_dict(lang="ar")

    def get_edge_cases(self) -> List[Dict[str, Any]]:
        """Get prompts tagged with 'edge'."""
        return self.get_prompts_as_dict(tags=["edge"])

    def get_xss_payloads(self) -> List[Dict[str, Any]]:
        """Get XSS security payloads."""
        payloads = self.get_security_payloads(p_type="xss")
        return [{"id": p.id, "p": p.p, "t": p.t} for p in payloads]

    def get_prompt_injections(self) -> List[Dict[str, Any]]:
        """Get prompt injection payloads."""
        payloads = self.get_security_payloads(p_type="prompt")
        return [{"id": p.id, "p": p.p, "t": p.t} for p in payloads]

    def get_sql_injections(self) -> List[Dict[str, Any]]:
        """Get SQL injection payloads."""
        payloads = self.get_security_payloads(p_type="sql")
        return [{"id": p.id, "p": p.p, "t": p.t} for p in payloads]