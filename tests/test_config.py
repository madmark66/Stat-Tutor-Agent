"""Tests for centralized configuration and shared UI configuration usage."""

import ast
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from stat_tutor.config import (
    DEFAULT_DB_PATH,
    DEFAULT_ENV_PATH,
    DEFAULT_MODEL_NAME,
    DEFAULT_STUDENT_ID,
    DEFAULT_TOPIC,
    PROJECT_ROOT,
    TutorConfig,
)


class TestTutorConfig(unittest.TestCase):
    def test_project_root_and_default_paths_are_absolute(self) -> None:
        expected_root = Path("/workspaces/Stat-Tutor-Agent").resolve()
        self.assertEqual(PROJECT_ROOT, expected_root)
        self.assertTrue(Path(DEFAULT_DB_PATH).is_absolute())
        self.assertEqual(
            Path(DEFAULT_DB_PATH),
            expected_root / "tutor_memory.db",
        )
        self.assertTrue(Path(DEFAULT_ENV_PATH).is_absolute())
        self.assertEqual(
            Path(DEFAULT_ENV_PATH),
            expected_root / ".env",
        )

    def test_defaults_when_only_api_key_set(self) -> None:
        with patch.dict(
            os.environ,
            {
                "GEMINI_API_KEY": "dummy_key_123",
                "GEMINI_MODEL": "",
                "DEFAULT_STUDENT_ID": "",
                "DEFAULT_TOPIC": "",
                "STAT_TUTOR_DB_PATH": "",
            },
            clear=False,
        ):
            cfg = TutorConfig.from_env()
            self.assertEqual(cfg.api_key, "dummy_key_123")
            self.assertEqual(cfg.model_name, DEFAULT_MODEL_NAME)
            self.assertEqual(cfg.model_name, "gemini-3.5-flash")
            self.assertEqual(cfg.default_student_id, DEFAULT_STUDENT_ID)
            self.assertEqual(cfg.default_student_id, "user_raw_01")
            self.assertEqual(cfg.default_topic, DEFAULT_TOPIC)
            self.assertEqual(cfg.db_path, DEFAULT_DB_PATH)

    def test_db_path_identical_across_different_working_directories(self) -> None:
        original_cwd = os.getcwd()
        try:
            with patch.dict(
                os.environ,
                {
                    "GEMINI_API_KEY": "dummy_key_cwd",
                    "STAT_TUTOR_DB_PATH": "",
                },
                clear=False,
            ):
                os.chdir(PROJECT_ROOT)
                cfg_from_root = TutorConfig.from_env()

                with tempfile.TemporaryDirectory() as other_dir:
                    os.chdir(other_dir)
                    cfg_from_other = TutorConfig.from_env()

                self.assertEqual(cfg_from_root.db_path, DEFAULT_DB_PATH)
                self.assertEqual(cfg_from_other.db_path, DEFAULT_DB_PATH)
                self.assertEqual(cfg_from_root.db_path, cfg_from_other.db_path)
        finally:
            os.chdir(original_cwd)

    def test_dotenv_loaded_from_fixed_project_root_regardless_of_cwd(self) -> None:
        original_cwd = os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as other_dir:
                os.chdir(other_dir)
                with patch("stat_tutor.config.load_dotenv") as mock_dotenv:
                    with patch.dict(
                        os.environ,
                        {"GEMINI_API_KEY": "dummy_key_env"},
                        clear=False,
                    ):
                        TutorConfig.from_env()
                        mock_dotenv.assert_called_once_with(
                            dotenv_path=str(PROJECT_ROOT / ".env"),
                            override=False,
                        )
        finally:
            os.chdir(original_cwd)

    def test_absolute_db_path_env_override_is_respected(self) -> None:
        with patch.dict(
            os.environ,
            {
                "GEMINI_API_KEY": "custom_key",
                "GEMINI_MODEL": "gemini-2.5-flash",
                "DEFAULT_STUDENT_ID": "student_custom_99",
                "DEFAULT_TOPIC": "信賴區間",
                "STAT_TUTOR_DB_PATH": "/tmp/custom_tutor.db",
            },
            clear=False,
        ):
            cfg = TutorConfig.from_env()
            self.assertEqual(cfg.model_name, "gemini-2.5-flash")
            self.assertEqual(cfg.default_student_id, "student_custom_99")
            self.assertEqual(cfg.default_topic, "信賴區間")
            self.assertEqual(cfg.db_path, str(Path("/tmp/custom_tutor.db").resolve()))

    def test_relative_db_path_env_override_resolved_against_project_root(self) -> None:
        original_cwd = os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as other_dir:
                os.chdir(other_dir)
                with patch.dict(
                    os.environ,
                    {
                        "GEMINI_API_KEY": "custom_key",
                        "STAT_TUTOR_DB_PATH": "data/sub_memory.db",
                    },
                    clear=False,
                ):
                    cfg = TutorConfig.from_env()
                    expected_path = str((PROJECT_ROOT / "data" / "sub_memory.db").resolve())
                    self.assertEqual(cfg.db_path, expected_path)
        finally:
            os.chdir(original_cwd)

    def test_missing_api_key_raises_runtime_error(self) -> None:
        with patch("stat_tutor.config.load_dotenv"):
            with patch.dict(os.environ, {"GEMINI_API_KEY": "   "}, clear=False):
                with self.assertRaises(RuntimeError):
                    TutorConfig.from_env()

    def test_both_ui_entrypoints_use_tutor_config_and_not_hardcoded_ids(self) -> None:
        for filename in ("app.py", "cli_tutor.py"):
            file_path = PROJECT_ROOT / filename
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()

            tree = ast.parse(source)
            self.assertIn("TutorConfig.from_env()", source)
            # Ensure neither UI file hardcodes old student IDs or model names
            self.assertNotIn("user_kashiwa_01", source)
            self.assertNotIn("user_raw_01", source)
            self.assertNotIn("gemini-3.8-flash", source)
            self.assertNotIn("gemini-3.5-flash", source)
            self.assertIsNotNone(tree)


if __name__ == "__main__":
    unittest.main()
