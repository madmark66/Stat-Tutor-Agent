"""CLI entry point for Stat-Tutor powered by the shared dual-brain core."""

import json
from typing import Any
from stat_tutor.config import TutorConfig
from stat_tutor.memory_store import LearningMemoryStore
from stat_tutor.models import LLMCommunicationError, ToolExecutionLog
from stat_tutor.tutor_core import TutorOrchestrator


def _print_thinking(_round_number: int) -> None:
    print("\n⏳ Gemini AI 思考中...")


def _print_tool_execution(log: ToolExecutionLog) -> None:
    print(f"\n🔧 [Function Calling] {log.function_name}")
    print(f"   參數：{json.dumps(log.args, ensure_ascii=False)}")
    if log.function_name == "calculate_sample_variance" and not log.is_error:
        print(f"⚙️ [Python 計算引擎] 數據: {log.args.get('numbers')}")
    elif log.function_name == "record_student_mistake" and not log.is_error:
        print(
            f"💾 [記憶寫入] {log.args.get('topic')}: {log.args.get('mistake_note')}"
        )
    print(f"   結果：{json.dumps(log.result, ensure_ascii=False)}")


def main() -> None:
    config = TutorConfig.from_env()
    memory_store = LearningMemoryStore(db_path=config.db_path)
    orchestrator = TutorOrchestrator(
        config=config,
        memory_store=memory_store,
    )

    student_id = config.default_student_id
    topic = config.default_topic
    past_mistake = memory_store.get_last_mistake(student_id, topic)

    history: list[dict[str, Any]] = []

    print("=" * 55)
    print("🧠 Stat-Tutor AI 統計家教啟動")
    print("=" * 55)

    print(f"模型：{config.model_name}")
    print(f"學生編號：{student_id}")
    print(f"學習主題：{topic}")
    print(f"過去錯誤紀錄：{past_mistake}")

    print("\n🤖 家教：")
    print("你好！今天我們來學習樣本變異數。")
    print("請提供一組數據，或直接提出你的問題。")
    print("\n輸入 quit 或 exit 結束對話。")
    print("輸入 reset 清除本次對話歷史。")

    while True:
        try:
            user_input = input("\n🧑 學生：").strip()

            if user_input.lower() in ["quit", "exit"]:
                print("\n👋 統計家教結束，再見！")
                break

            if user_input.lower() == "reset":
                history.clear()
                print("\n🔄 本次對話歷史已清除。")
                continue

            if not user_input:
                continue

            turn_result = orchestrator.process_message(
                user_input=user_input,
                history=history,
                student_id=student_id,
                topic=topic,
                on_thinking=_print_thinking,
                on_tool_call=_print_tool_execution,
            )

            print("\n🤖 家教：")
            print(turn_result.reply_text)

        except KeyboardInterrupt:
            print("\n\n👋 使用者中斷程式。")
            break

        except LLMCommunicationError as exc:
            print(f"\n❌ {exc}")

        except Exception as exc:
            print(f"\n❌ 系統錯誤：{exc}")


if __name__ == "__main__":
    main()
