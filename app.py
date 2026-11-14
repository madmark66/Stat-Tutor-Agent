"""Streamlit Web UI entry point for Stat-Tutor powered by the shared dual-brain core."""

import time
import streamlit as st
from stat_tutor.config import TutorConfig
from stat_tutor.memory_store import LearningMemoryStore
from stat_tutor.models import LLMCommunicationError
from stat_tutor.tutor_core import TutorOrchestrator

st.set_page_config(page_title="雙腦 AI 家教", page_icon="🧠")
st.title("🧠 雙腦協作 AI 家教")

# ==========================================
# 1. 載入集中化設定與雙腦核心
# ==========================================
try:
    config = TutorConfig.from_env()
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()

memory_store = LearningMemoryStore(db_path=config.db_path)
orchestrator = TutorOrchestrator(
    config=config,
    memory_store=memory_store,
)

# ==========================================
# 2. 側邊欄：學生編號、主題與學習記憶狀態
# ==========================================
if "active_student_id" not in st.session_state:
    st.session_state.active_student_id = config.default_student_id

if "active_topic" not in st.session_state:
    st.session_state.active_topic = config.default_topic

with st.sidebar:
    st.header("⚙️ 學習設定與記憶庫")
    st.caption(f"目前模型：`{config.model_name}`")

    new_student_id = st.text_input(
        "學生編號 (Student ID)",
        value=st.session_state.active_student_id,
    ).strip()
    new_topic = st.text_input(
        "目前學習主題",
        value=st.session_state.active_topic,
    ).strip()

    if new_student_id and new_student_id != st.session_state.active_student_id:
        st.session_state.active_student_id = new_student_id
        st.session_state.pop("ui_messages", None)
        st.session_state.pop("llm_history", None)
        st.rerun()

    if new_topic and new_topic != st.session_state.active_topic:
        st.session_state.active_topic = new_topic

    current_mistake = memory_store.get_last_mistake(
        st.session_state.active_student_id,
        st.session_state.active_topic,
    )
    st.markdown("**目前主題錯誤紀錄：**")
    st.info(current_mistake)

    all_records = memory_store.list_student_records(
        st.session_state.active_student_id
    )
    if all_records:
        with st.expander("📚 該學生所有主題記憶", expanded=False):
            for rec in all_records:
                st.markdown(
                    f"- **{rec['topic']}**：{rec['last_mistake_note']} "
                    f"*(更新：{rec['updated_at'] or '未記錄'})*"
                )

    if st.button("🔄 清除本次對話歷史"):
        st.session_state.pop("ui_messages", None)
        st.session_state.pop("llm_history", None)
        st.rerun()

# ==========================================
# 3. 對話狀態管理與渲染
# ==========================================
if "ui_messages" not in st.session_state:
    st.session_state.ui_messages = [
        {
            "role": "assistant",
            "content": "你好！請給我一組數據，我們來算變異數。",
            "tool_logs": [],
        }
    ]

if "llm_history" not in st.session_state:
    st.session_state.llm_history = []

for msg in st.session_state.ui_messages:
    with st.chat_message(msg["role"]):
        if msg.get("tool_logs"):
            with st.expander("🔧 左腦工具與記憶呼叫紀錄", expanded=False):
                for log in msg["tool_logs"]:
                    st.json(
                        {
                            "function": log["function_name"],
                            "args": log["args"],
                            "result": log["result"],
                            "is_error": log["is_error"],
                        }
                    )
        st.markdown(msg["content"])

if prompt := st.chat_input("輸入數據或回覆..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.ui_messages.append(
        {"role": "user", "content": prompt, "tool_logs": []}
    )

    with st.chat_message("assistant"):
        with st.spinner("雙腦協作計算與思考中..."):
            try:
                turn_result = orchestrator.process_message(
                    user_input=prompt,
                    history=st.session_state.llm_history,
                    student_id=st.session_state.active_student_id,
                    topic=st.session_state.active_topic,
                )

                serialized_logs = [
                    {
                        "function_name": log.function_name,
                        "args": log.args,
                        "result": log.result,
                        "is_error": log.is_error,
                    }
                    for log in turn_result.tool_logs
                ]

                if serialized_logs:
                    with st.expander("🔧 左腦工具與記憶呼叫紀錄", expanded=False):
                        for log in serialized_logs:
                            st.json(
                                {
                                    "function": log["function_name"],
                                    "args": log["args"],
                                    "result": log["result"],
                                    "is_error": log["is_error"],
                                }
                            )

                placeholder = st.empty()

                def stream_text():
                    for char in turn_result.reply_text:
                        yield char
                        time.sleep(0.005)

                full_response = placeholder.write_stream(stream_text)

                st.session_state.ui_messages.append(
                    {
                        "role": "assistant",
                        "content": full_response,
                        "tool_logs": serialized_logs,
                    }
                )

            except LLMCommunicationError as exc:
                st.error(f"連線錯誤: {exc}")
            except Exception as exc:
                st.error(f"發生錯誤: {exc}")
