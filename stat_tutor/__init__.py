"""Stat-Tutor dual-brain core package."""

from stat_tutor.config import (
    DEFAULT_DB_PATH,
    DEFAULT_MODEL_NAME,
    DEFAULT_STUDENT_ID,
    DEFAULT_TOPIC,
    TutorConfig,
)
from stat_tutor.gemini_client import GeminiClient
from stat_tutor.memory_store import LearningMemoryStore
from stat_tutor.models import (
    FunctionCallItem,
    LLMCommunicationError,
    ToolExecutionLog,
    TutorTurnResult,
)
from stat_tutor.stat_engine import calculate_sample_variance
from stat_tutor.tool_manager import ToolManager
from stat_tutor.tutor_core import TutorOrchestrator

__all__ = [
    "DEFAULT_DB_PATH",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_STUDENT_ID",
    "DEFAULT_TOPIC",
    "FunctionCallItem",
    "GeminiClient",
    "LLMCommunicationError",
    "LearningMemoryStore",
    "ToolExecutionLog",
    "ToolManager",
    "TutorConfig",
    "TutorOrchestrator",
    "TutorTurnResult",
    "calculate_sample_variance",
]
