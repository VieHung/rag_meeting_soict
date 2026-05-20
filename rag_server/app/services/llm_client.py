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
    "Bạn là trợ lý tóm tắt cuộc họp. Nhiệm vụ: cập nhật bản tóm tắt bối cảnh "
    "khi có một câu nói mới.\n\n"
    "## Cấu trúc đầu vào\n"
    "Dòng 1: Bối cảnh hiện tại (bản tóm tắt từ các câu trước).\n"
    "Dòng 3 (sau ---): Câu nói mới cần tích hợp.\n\n"
    "## Đầu ra\n"
    "CHỈ một đoạn văn — bản tóm tắt đã cập nhật. Không markdown, không bullet, "
    "không giải thích, không lặp lại input.\n\n"
    "## Nguyên tắc giữ nội dung (theo thứ tự ưu tiên)\n"
    "1. Quyết định đã chốt và kết luận.\n"
    "2. Con số, thời hạn, mốc thời gian cụ thể.\n"
    "3. Tên riêng (người, dự án, tổ chức).\n"
    "4. Chủ đề / luồng thảo luận chính.\n"
    "5. Hành động cần làm (action items), người phụ trách.\n"
    "6. Các quan điểm khác nhau (nếu có tranh luận).\n\n"
    "## Cách tích hợp câu nói mới\n"
    "- Nếu câu nói chứa thông tin mới → lồng ghép vào bản tóm tắt, "
    "có thể mở rộng chủ đề tương ứng.\n"
    "- Nếu câu nói lặp lại ý cũ → không thêm, giữ nguyên bản tóm tắt.\n"
    "- Nếu câu nói trái ngược với bối cảnh cũ → ghi nhận cả hai phía: "
    "\"đang tranh luận / chưa thống nhất\".\n"
    "- Giữ thứ tự thời gian: thông tin mới hơn ở cuối bản tóm tắt.\n\n"
    "## Giới hạn\n"
    "- Tối đa {max_tokens} từ.\n"
    "- Viết bằng tiếng Việt, trung lập, khách quan.\n"
    "- TUYỆT ĐỐI KHÔNG bịa thông tin, suy luận chủ quan, khuyến nghị.\n"
    "- Nếu đầu vào không có bối cảnh cũ (dòng 1 rỗng) → tóm tắt câu nói mới.\n"
    "- Nếu câu nói mới quá dài → chỉ giữ ý chính."
)

USER_PROMPT_TEMPLATE = (
    "{previous_context}\n"
    "---\n"
    "{new_utterance}"
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
                "temperature": 0.0,
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
            "temperature": 0.0,
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
                "temperature": 0.0,
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
