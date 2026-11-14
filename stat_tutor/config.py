"""Centralized configuration for Stat-Tutor."""

from dataclasses import dataclass
import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_PATH = str(PROJECT_ROOT / ".env")
DEFAULT_MODEL_NAME = "gemini-3.5-flash"
DEFAULT_STUDENT_ID = "user_raw_01"
DEFAULT_TOPIC = "樣本變異數"
DEFAULT_DB_PATH = str(PROJECT_ROOT / "tutor_memory.db")
DEFAULT_REQUEST_TIMEOUT = 45
DEFAULT_MAX_TOOL_ROUNDS = 8


def resolve_db_path(raw_db_path: str | Path | None) -> str:
    """
    Resolve database path independently of the Current Working Directory (CWD):
    - If empty or None, return DEFAULT_DB_PATH (<PROJECT_ROOT>/tutor_memory.db).
    - If absolute path, preserve and normalize it.
    - If relative path, resolve it relative to PROJECT_ROOT.
    """
    if raw_db_path is None:
        return DEFAULT_DB_PATH

    cleaned = str(raw_db_path).strip()
    if not cleaned:
        return DEFAULT_DB_PATH

    candidate = Path(cleaned)
    if candidate.is_absolute():
        return str(candidate.resolve())

    return str((PROJECT_ROOT / candidate).resolve())


@dataclass(frozen=True)
class TutorConfig:
    """Immutable configuration object shared across CLI, Streamlit, and Core."""

    api_key: str
    model_name: str = DEFAULT_MODEL_NAME
    default_student_id: str = DEFAULT_STUDENT_ID
    default_topic: str = DEFAULT_TOPIC
    db_path: str = DEFAULT_DB_PATH
    request_timeout: int = DEFAULT_REQUEST_TIMEOUT
    max_tool_rounds: int = DEFAULT_MAX_TOOL_ROUNDS

    @classmethod
    def from_env(
        cls,
        env_file: str | Path | None = None,
        require_api_key: bool = True,
    ) -> "TutorConfig":
        """Load configuration from fixed project .env and environment variables."""
        target_env = str(env_file) if env_file else DEFAULT_ENV_PATH
        load_dotenv(dotenv_path=target_env, override=False)

        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if require_api_key and not api_key:
            raise RuntimeError(
                "找不到 GEMINI_API_KEY，請檢查 Codespaces Secrets 或本機 .env"
            )

        model_name = (
            os.environ.get("GEMINI_MODEL", "").strip() or DEFAULT_MODEL_NAME
        )
        default_student_id = (
            os.environ.get("DEFAULT_STUDENT_ID", "").strip()
            or DEFAULT_STUDENT_ID
        )
        default_topic = (
            os.environ.get("DEFAULT_TOPIC", "").strip() or DEFAULT_TOPIC
        )
        raw_db_path = os.environ.get("STAT_TUTOR_DB_PATH", "").strip()
        db_path = resolve_db_path(raw_db_path)

        return cls(
            api_key=api_key,
            model_name=model_name,
            default_student_id=default_student_id,
            default_topic=default_topic,
            db_path=db_path,
        )
