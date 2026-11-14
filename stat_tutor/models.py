"""Shared data structures and exception types for Stat-Tutor."""

from dataclasses import dataclass, field
from typing import Any


class LLMCommunicationError(RuntimeError):
    """Raised when the LLM communication adapter encounters network, timeout, or API errors."""


@dataclass
class FunctionCallItem:
    """Represents a single function call requested by the LLM."""

    name: str
    args: dict[str, Any]
    call_id: str | None = None


@dataclass
class ToolExecutionLog:
    """Records the execution details and result of a single tool invocation."""

    function_name: str
    args: dict[str, Any]
    result: dict[str, Any]
    is_error: bool = False


@dataclass
class TutorTurnResult:
    """Represents the completed outcome of a single user turn."""

    reply_text: str
    tool_logs: list[ToolExecutionLog] = field(default_factory=list)
