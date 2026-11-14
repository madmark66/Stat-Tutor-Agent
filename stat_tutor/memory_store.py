"""SQLite persistence layer for student learning memory."""

import sqlite3
from typing import Any

MAX_STUDENT_ID_LENGTH = 100
MAX_TOPIC_LENGTH = 100
MAX_MISTAKE_NOTE_LENGTH = 1000


class LearningMemoryStore:
    """Manages reading and writing student learning records in SQLite."""

    def __init__(self, db_path: str = "tutor_memory.db") -> None:
        if not isinstance(db_path, str) or not db_path.strip():
            raise ValueError("資料庫路徑不可為空")
        self.db_path = db_path.strip()
        self.init_db()

    def init_db(self) -> None:
        """Ensure mastery_profile table and updated_at column exist without altering existing records."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS mastery_profile (
                    student_id TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    last_mistake_note TEXT,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (student_id, topic)
                )
            """)

            columns = {
                row[1]
                for row in conn.execute("PRAGMA table_info(mastery_profile)")
            }
            if "updated_at" not in columns:
                conn.execute("""
                    ALTER TABLE mastery_profile
                    ADD COLUMN updated_at TEXT
                """)

    def get_last_mistake(self, student_id: str, topic: str) -> str:
        """Retrieve the last recorded mistake note for a student and topic, or '無' if none."""
        if not isinstance(student_id, str) or not student_id.strip():
            return "無"
        if not isinstance(topic, str) or not topic.strip():
            return "無"

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT last_mistake_note
                FROM mastery_profile
                WHERE student_id = ? AND topic = ?
                """,
                (student_id.strip(), topic.strip()),
            )
            row = cursor.fetchone()
            return row[0] if row and row[0] else "無"

    def list_student_records(self, student_id: str) -> list[dict[str, Any]]:
        """Return all mastery_profile records for a given student_id (read-only helper for UI)."""
        if not isinstance(student_id, str) or not student_id.strip():
            return []

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT topic, last_mistake_note, updated_at
                FROM mastery_profile
                WHERE student_id = ?
                ORDER BY updated_at DESC, topic ASC
                """,
                (student_id.strip(),),
            )
            return [
                {
                    "topic": row[0],
                    "last_mistake_note": row[1],
                    "updated_at": row[2],
                }
                for row in cursor.fetchall()
            ]

    def record_student_mistake(
        self,
        student_id: str,
        topic: str,
        mistake_note: str,
    ) -> dict[str, str]:
        """
        Validate and upsert a student's conceptual mistake into mastery_profile.

        Explicit write rules:
        1. student_id, topic, and mistake_note must be non-empty strings.
        2. Length limits are enforced to prevent database pollution.
        """
        if not isinstance(student_id, str) or not student_id.strip():
            raise ValueError("學生編號不可為空")
        if not isinstance(topic, str) or not topic.strip():
            raise ValueError("主題及錯誤紀錄不可為空")
        if not isinstance(mistake_note, str) or not mistake_note.strip():
            raise ValueError("主題及錯誤紀錄不可為空")

        clean_student_id = student_id.strip()
        clean_topic = topic.strip()
        clean_note = mistake_note.strip()

        if len(clean_student_id) > MAX_STUDENT_ID_LENGTH:
            raise ValueError(f"學生編號長度不可超過 {MAX_STUDENT_ID_LENGTH} 字元")
        if len(clean_topic) > MAX_TOPIC_LENGTH:
            raise ValueError(f"主題長度不可超過 {MAX_TOPIC_LENGTH} 字元")
        if len(clean_note) > MAX_MISTAKE_NOTE_LENGTH:
            raise ValueError(f"錯誤紀錄長度不可超過 {MAX_MISTAKE_NOTE_LENGTH} 字元")

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO mastery_profile
                    (student_id, topic, last_mistake_note, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(student_id, topic)
                DO UPDATE SET
                    last_mistake_note = excluded.last_mistake_note,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (clean_student_id, clean_topic, clean_note),
            )

        return {
            "status": "success",
            "message": "學習錯誤紀錄已儲存",
        }
