"""Abstraction cho LLM dùng để build context tóm tắt.

Hỗ trợ ban đầu:
- ollama  : POST {base_url}/api/chat
- gemini  : POST https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent
- openai  : POST {base_url}/chat/completions  (compatible: OpenAI / vLLM / LocalAI)
- none    : no-op stub (luôn trả "") — dùng khi muốn tắt build context

Provider được chọn qua `settings.llm_provider`.
"""
from __future__ import annotations

import asyncio
from typing import Optional

import httpx

from app.config import settings


SYSTEM_PROMPT_TEMPLATE = (
    "Bạn là trợ lý tóm tắt hội thoại cuộc họp. Nhiệm vụ: cập nhật bản tóm tắt "
    "bối cảnh cuộc họp khi có một câu nói mới. Bản tóm tắt phải:\n"
    "- Ngắn gọn, tối đa khoảng {max_tokens} tokens.\n"
    "- Giữ các quyết định, con số, tên riêng, chủ đề đang thảo luận.\n"
    "- Viết bằng tiếng Việt, văn phong trung lập, không bịa thông tin.\n"
    "- Chỉ trả về bản tóm tắt, không thêm lời dẫn."
)

USER_PROMPT_TEMPLATE = (
    "[Bối cảnh hiện tại]\n{previous_context}\n\n"
    "[Câu nói mới cần tích hợp vào bối cảnh]\n{new_utterance}\n\n"
    "Hãy cập nhật bản tóm tắt bối cảnh."
)


class LLMError(RuntimeError):
    """Bao đóng lỗi gọi LLM (timeout / HTTP / parse)."""


class LLMClient:
    """LLM client trừu tượng cho ContextBuilder.

    Mọi provider implement chung phương thức `summarize(previous_context, new_utterance) -> str`.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        self.provider = (provider or settings.llm_provider or "none").lower()
        self.model = model or settings.llm_model
        self.base_url = (base_url or settings.llm_base_url or "").rstrip("/")
        self.api_key = api_key if api_key is not None else settings.llm_api_key
        self.timeout = timeout or settings.context_timeout_seconds
        self.max_tokens = max_tokens or settings.context_max_tokens
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:
                pass
            self._client = None

    # ----- public API ------------------------------------------------------

    async def summarize(self, previous_context: str, new_utterance: str) -> str:
        """Gọi LLM cập nhật context. Raises LLMError on failure."""
        if self.provider == "none":
            return ""

        system = SYSTEM_PROMPT_TEMPLATE.format(max_tokens=self.max_tokens)
        user = USER_PROMPT_TEMPLATE.format(
            previous_context=(previous_context or "(chưa có bối cảnh)"),
            new_utterance=new_utterance,
        )

        try:
            if self.provider == "ollama":
                text = await self._call_ollama(system, user)
            elif self.provider == "gemini":
                text = await self._call_gemini(system, user)
            elif self.provider == "openai":
                text = await self._call_openai(system, user)
            else:
                raise LLMError(f"Unknown LLM provider: {self.provider}")
        except httpx.TimeoutException as e:
            raise LLMError(f"LLM timeout after {self.timeout}s") from e
        except httpx.HTTPError as e:
            raise LLMError(f"LLM HTTP error: {e}") from e

        text = (text or "").strip()
        return self._truncate(text)

    # ----- providers -------------------------------------------------------

    async def _call_ollama(self, system: str, user: str) -> str:
        client = await self._get_client()
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {
                # Ollama dùng `num_predict` để giới hạn output tokens.
                "num_predict": self.max_tokens,
                "temperature": 0.2,
            },
        }
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        # /api/chat trả {"message": {"role": "assistant", "content": "..."}}
        msg = data.get("message") or {}
        content = msg.get("content")
        if not content:
            # Fallback (vài bản Ollama trả "response")
            content = data.get("response", "")
        return content or ""

    async def _call_openai(self, system: str, user: str) -> str:
        client = await self._get_client()
        url = f"{self.base_url}/chat/completions" if self.base_url else "https://api.openai.com/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            return ""
        return (choices[0].get("message") or {}).get("content", "") or ""

    async def _call_gemini(self, system: str, user: str) -> str:
        client = await self._get_client()
        # Gemini REST endpoint
        base = self.base_url or "https://generativelanguage.googleapis.com/v1beta"
        url = f"{base}/models/{self.model}:generateContent"
        params = {}
        if self.api_key:
            params["key"] = self.api_key
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": self.max_tokens,
            },
        }
        resp = await client.post(url, params=params, json=payload)
        resp.raise_for_status()
        data = resp.json()
        candidates = data.get("candidates") or []
        if not candidates:
            return ""
        parts = ((candidates[0].get("content") or {}).get("parts")) or []
        return "".join(p.get("text", "") for p in parts)

    # ----- helpers ---------------------------------------------------------

    def _truncate(self, text: str) -> str:
        """Cắt sơ bộ theo word count để bảo vệ độ dài context.

        Token-based truncation chính xác cần tokenizer riêng cho từng model,
        ở đây ta dùng heuristic ~ 1.3 words/token cho tiếng Việt.
        """
        if not text:
            return text
        max_words = max(1, int(self.max_tokens * 1.3))
        words = text.split()
        if len(words) <= max_words:
            return text
        return " ".join(words[:max_words]).rstrip()


# Singleton accessor (để dependency có thể chia sẻ httpx client)
_llm_singleton: Optional[LLMClient] = None
_lock = asyncio.Lock()


def get_llm_client() -> LLMClient:
    global _llm_singleton
    if _llm_singleton is None:
        _llm_singleton = LLMClient()
    return _llm_singleton


async def shutdown_llm_client() -> None:
    global _llm_singleton
    if _llm_singleton is not None:
        await _llm_singleton.aclose()
        _llm_singleton = None
