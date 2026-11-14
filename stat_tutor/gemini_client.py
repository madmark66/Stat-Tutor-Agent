"""LLM communication adapter for Google Gemini REST API."""

from typing import Any
import requests
from stat_tutor.models import LLMCommunicationError


class GeminiClient:
    """
    Replaceable communication adapter for Gemini Function Calling.
    Initially uses the Gemini v1beta REST endpoint; can be swapped for google-genai later
    without changing TutorOrchestrator.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str,
        timeout: int = 45,
    ) -> None:
        if not api_key or not api_key.strip():
            raise ValueError("API Key 不可為空")
        if not model_name or not model_name.strip():
            raise ValueError("模型名稱不可為空")

        self.api_key = api_key.strip()
        self.model_name = model_name.strip()
        self.timeout = timeout
        self.url = (
            f"https://generativelanguage.googleapis.com/"
            f"v1beta/models/{self.model_name}:generateContent"
        )

    def generate_content(
        self,
        system_instruction: str,
        contents: list[dict[str, Any]],
        tools_schema: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Send a generateContent request to Gemini and return the candidate content dict.
        Raises LLMCommunicationError on network, timeout, HTTP, or malformed response errors.
        """
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }
        payload = {
            "systemInstruction": {
                "parts": [{"text": system_instruction}],
            },
            "contents": contents,
            "tools": tools_schema,
        }

        try:
            response = requests.post(
                self.url,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
        except requests.exceptions.Timeout as exc:
            raise LLMCommunicationError(
                "API 連線逾時，請檢查網路或稍後重試。"
            ) from exc
        except requests.exceptions.ConnectionError as exc:
            raise LLMCommunicationError(
                "無法連接 Gemini API，請檢查網路連線。"
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise LLMCommunicationError(f"HTTP 請求錯誤：{exc}") from exc

        if not response.ok:
            try:
                error_info = response.json().get("error", {})
                message = error_info.get("message", response.text)
                code = error_info.get("code", response.status_code)
            except ValueError:
                message = response.text
                code = response.status_code

            raise LLMCommunicationError(f"Gemini API 錯誤 {code}: {message}")

        try:
            data = response.json()
        except ValueError as exc:
            raise LLMCommunicationError("Gemini API 回傳非合法的 JSON 格式") from exc

        candidates = data.get("candidates", [])
        if not candidates:
            raise LLMCommunicationError(
                "Gemini 沒有回傳候選答案，請檢查安全過濾或 API 回應"
            )

        candidate = candidates[0]
        content = candidate.get("content")
        if not content or not isinstance(content, dict):
            raise LLMCommunicationError(
                f"Gemini 沒有回傳內容，finishReason={candidate.get('finishReason')}"
            )

        return content
