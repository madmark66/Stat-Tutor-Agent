"""Unit tests for LearningMemoryStore and existing SQLite database preservation."""

import os
import sqlite3
import tempfile
import unittest
from stat_tutor.memory_store import (
    LearningMemoryStore,
    MAX_MISTAKE_NOTE_LENGTH,
    MAX_TOPIC_LENGTH,
)


class TestMemoryStore(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_memory.db")
        self.store = LearningMemoryStore(db_path=self.db_path)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_get_last_mistake_when_empty_returns_wu(self) -> None:
        self.assertEqual(
            self.store.get_last_mistake("user_test", "樣本變異數"),
            "無",
        )

    def test_record_and_upsert_student_mistake(self) -> None:
        res1 = self.store.record_student_mistake(
            student_id="user_test",
            topic="樣本變異數",
            mistake_note="忘記將離均差平方",
        )
        self.assertEqual(res1["status"], "success")
        self.assertEqual(
            self.store.get_last_mistake("user_test", "樣本變異數"),
            "忘記將離均差平方",
        )

        # Upsert with new mistake on same (student_id, topic)
        res2 = self.store.record_student_mistake(
            student_id="user_test",
            topic="樣本變異數",
            mistake_note="分母誤除以 n 而非 n-1",
        )
        self.assertEqual(res2["status"], "success")
        self.assertEqual(
            self.store.get_last_mistake("user_test", "樣本變異數"),
            "分母誤除以 n 而非 n-1",
        )

        records = self.store.list_student_records("user_test")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["topic"], "樣本變異數")
        self.assertIsNotNone(records[0]["updated_at"])

    def test_legacy_table_migration_adds_updated_at(self) -> None:
        legacy_db = os.path.join(self.temp_dir.name, "legacy.db")
        with sqlite3.connect(legacy_db) as conn:
            conn.execute(
                "CREATE TABLE mastery_profile ("
                "student_id TEXT, topic TEXT, last_mistake_note TEXT, "
                "PRIMARY KEY (student_id, topic))"
            )
            conn.execute(
                "INSERT INTO mastery_profile VALUES (?, ?, ?)",
                ("old_user", "舊主題", "舊錯誤紀錄"),
            )

        migrated_store = LearningMemoryStore(db_path=legacy_db)
        self.assertEqual(
            migrated_store.get_last_mistake("old_user", "舊主題"),
            "舊錯誤紀錄",
        )
        migrated_store.record_student_mistake("old_user", "舊主題", "更新後錯誤")
        self.assertEqual(
            migrated_store.get_last_mistake("old_user", "舊主題"),
            "更新後錯誤",
        )

    def test_explicit_write_validation_rules(self) -> None:
        with self.assertRaises(ValueError):
            self.store.record_student_mistake("", "主題", "筆記")
        with self.assertRaises(ValueError):
            self.store.record_student_mistake("user", "   ", "筆記")
        with self.assertRaises(ValueError):
            self.store.record_student_mistake("user", "主題", "")
        with self.assertRaises(ValueError):
            self.store.record_student_mistake(
                "user", "T" * (MAX_TOPIC_LENGTH + 1), "筆記"
            )
        with self.assertRaises(ValueError):
            self.store.record_student_mistake(
                "user", "主題", "N" * (MAX_MISTAKE_NOTE_LENGTH + 1)
            )

    def test_existing_production_db_records_remain_intact(self) -> None:
        prod_db = "/workspaces/Stat-Tutor-Agent/tutor_memory.db"
        if not os.path.exists(prod_db):
            self.skipTest("Production tutor_memory.db not present")

        with sqlite3.connect(prod_db) as conn:
            before_rows = conn.execute(
                "SELECT student_id, topic, last_mistake_note, updated_at "
                "FROM mastery_profile ORDER BY student_id, topic"
            ).fetchall()

        prod_store = LearningMemoryStore(db_path=prod_db)
        note = prod_store.get_last_mistake("user_raw_01", "樣本變異數與自由度")
        self.assertEqual(
            note,
            "學生認為計算樣本變異數時是直接除以個數 n，而非除以 n - 1。",
        )

        with sqlite3.connect(prod_db) as conn:
            after_rows = conn.execute(
                "SELECT student_id, topic, last_mistake_note, updated_at "
                "FROM mastery_profile ORDER BY student_id, topic"
            ).fetchall()

        self.assertEqual(before_rows, after_rows)


if __name__ == "__main__":
    unittest.main()
