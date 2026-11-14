"""Tool registration, strict argument validation, and execution dispatch."""

from typing import Any
from stat_tutor.memory_store import LearningMemoryStore
from stat_tutor.models import ToolExecutionLog
from stat_tutor.stat_engine import calculate_sample_variance


TOOLS_SCHEMA: list[dict[str, Any]] = [
    {
        "functionDeclarations": [
            {
                "name": "calculate_sample_variance",
                "description": (
                    "使用 Python 計算一組數字的樣本變異數（除以 n-1 之 variance_ddof1）"
                    "與母體變異數（除以 n 之 variance_ddof0）。"
                    "當學生提供數據或詢問變異數計算時必須使用。"
                ),
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "numbers": {
                            "type": "ARRAY",
                            "items": {"type": "NUMBER"},
                            "description": "要計算的數字列表（至少包含兩個有限數字）",
                        }
                    },
                    "required": ["numbers"],
                },
            },
            {
                "name": "record_student_mistake",
                "description": (
                    "當學生出現明確的統計觀念錯誤時，"
                    "記錄其錯誤主題與具體錯誤內容。"
                    "不要將一般提問當成錯誤。"
                ),
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "topic": {
                            "type": "STRING",
                            "description": "統計學習主題名稱",
                        },
                        "mistake_note": {
                            "type": "STRING",
                            "description": "學生具體觀念錯誤描述",
                        },
                    },
                    "required": ["topic", "mistake_note"],
                },
            },
        ]
    }
]


class ToolManager:
    """Validates LLM tool call arguments and dispatches to deterministic engines."""

    def __init__(self, memory_store: LearningMemoryStore) -> None:
        self.memory_store = memory_store

    def get_tools_schema(self) -> list[dict[str, Any]]:
        """Return the tool declarations schema for the LLM client."""
        return TOOLS_SCHEMA

    def execute_tool(
        self,
        function_name: Any,
        raw_args: Any,
        student_id: str,
    ) -> ToolExecutionLog:
        """
        Validate tool name and arguments strictly before dispatching execution.
        Never raises unhandled ValueError/TypeError on malformed LLM arguments;
        returns a structured ToolExecutionLog with is_error=True instead.
        """
        safe_name = str(function_name) if function_name is not None else ""
        if not isinstance(raw_args, dict):
            return ToolExecutionLog(
                function_name=safe_name,
                args={},
                result={
                    "status": "error",
                    "message": "工具參數格式錯誤：args 必須為 JSON 物件 (dict)",
                },
                is_error=True,
            )

        try:
            if safe_name == "calculate_sample_variance":
                if "numbers" not in raw_args:
                    raise ValueError("缺少必要參數：numbers")
                allowed_keys = {"numbers"}
                unexpected_keys = set(raw_args.keys()) - allowed_keys
                if unexpected_keys:
                    raise ValueError(f"不支援的多餘參數：{sorted(unexpected_keys)}")

                result = calculate_sample_variance(raw_args["numbers"])
                return ToolExecutionLog(
                    function_name=safe_name,
                    args=dict(raw_args),
                    result=result,
                    is_error=False,
                )

            if safe_name == "record_student_mistake":
                if "topic" not in raw_args or "mistake_note" not in raw_args:
                    raise ValueError("缺少必要參數：topic 與 mistake_note")
                allowed_keys = {"topic", "mistake_note"}
                unexpected_keys = set(raw_args.keys()) - allowed_keys
                if unexpected_keys:
                    raise ValueError(f"不支援的多餘參數：{sorted(unexpected_keys)}")

                topic = raw_args.get("topic")
                mistake_note = raw_args.get("mistake_note")
                if not isinstance(topic, str) or not isinstance(mistake_note, str):
                    raise ValueError("參數型別錯誤：topic 與 mistake_note 必須為字串")

                # Enforce session student_id from trusted context, not from LLM arguments.
                result = self.memory_store.record_student_mistake(
                    student_id=student_id,
                    topic=topic,
                    mistake_note=mistake_note,
                )
                return ToolExecutionLog(
                    function_name=safe_name,
                    args=dict(raw_args),
                    result=result,
                    is_error=False,
                )

            return ToolExecutionLog(
                function_name=safe_name,
                args=dict(raw_args),
                result={
                    "status": "error",
                    "message": f"未知工具：{safe_name}",
                },
                is_error=True,
            )

        except Exception as exc:
            return ToolExecutionLog(
                function_name=safe_name,
                args=dict(raw_args),
                result={
                    "status": "error",
                    "message": str(exc),
                },
                is_error=True,
            )
