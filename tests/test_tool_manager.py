"""Unit tests for ToolManager strict argument validation and dispatch."""

import os
import tempfile
import unittest
from stat_tutor.memory_store import LearningMemoryStore
from stat_tutor.tool_manager import ToolManager


class TestToolManager(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_tools.db")
        self.memory_store = LearningMemoryStore(db_path=self.db_path)
        self.tool_manager = ToolManager(memory_store=self.memory_store)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_valid_calculate_sample_variance_call(self) -> None:
        log = self.tool_manager.execute_tool(
            function_name="calculate_sample_variance",
            raw_args={"numbers": [10, 20, 30]},
            student_id="user_01",
        )
        self.assertFalse(log.is_error)
        self.assertEqual(log.result["status"], "success")
        self.assertAlmostEqual(log.result["variance_ddof1"], 100.0)
        self.assertAlmostEqual(log.result["variance_ddof0"], 200.0 / 3.0)

    def test_valid_record_student_mistake_enforces_session_student_id(self) -> None:
        log = self.tool_manager.execute_tool(
            function_name="record_student_mistake",
            raw_args={
                "topic": "樣本變異數",
                "mistake_note": "除以 n 而非 n-1",
            },
            student_id="trusted_student_01",
        )
        self.assertFalse(log.is_error)
        self.assertEqual(log.result["status"], "success")
        self.assertEqual(
            self.memory_store.get_last_mistake("trusted_student_01", "樣本變異數"),
            "除以 n 而非 n-1",
        )

    def test_malformed_or_unexpected_arguments_are_caught_safely(self) -> None:
        malformed_cases = [
            # Non-dict raw_args
            ("calculate_sample_variance", None),
            ("calculate_sample_variance", "[1, 2, 3]"),
            ("calculate_sample_variance", [1, 2, 3]),
            # Missing required arguments
            ("calculate_sample_variance", {}),
            ("record_student_mistake", {"topic": "樣本變異數"}),
            ("record_student_mistake", {"mistake_note": "某個錯誤"}),
            # Unexpected extra arguments (including attempted student_id spoofing)
            (
                "calculate_sample_variance",
                {"numbers": [1, 2, 3], "ddof": 0},
            ),
            (
                "record_student_mistake",
                {
                    "topic": "樣本變異數",
                    "mistake_note": "錯誤",
                    "student_id": "hacked_user",
                },
            ),
            # Wrong argument value types
            ("calculate_sample_variance", {"numbers": "1, 2, 3"}),
            ("calculate_sample_variance", {"numbers": [True, False]}),
            ("calculate_sample_variance", {"numbers": [5.0]}),
            ("record_student_mistake", {"topic": 123, "mistake_note": "錯誤"}),
            ("record_student_mistake", {"topic": "主題", "mistake_note": "   "}),
            # Unknown tool name
            ("drop_all_tables", {"confirm": True}),
        ]

        for func_name, bad_args in malformed_cases:
            with self.subTest(func_name=func_name, bad_args=bad_args):
                log = self.tool_manager.execute_tool(
                    function_name=func_name,
                    raw_args=bad_args,
                    student_id="trusted_student_01",
                )
                self.assertTrue(log.is_error)
                self.assertEqual(log.result["status"], "error")
                self.assertTrue(len(log.result.get("message", "")) > 0)


if __name__ == "__main__":
    unittest.main()
