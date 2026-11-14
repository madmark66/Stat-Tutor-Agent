import streamlit as st
import google.generativeai as genai
import numpy as np
import os
import sqlite3
import time
from dotenv import load_dotenv

# ==========================================
# 0. 環境設定
# ==========================================
load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY", "").strip()
if not api_key:
    st.error("請設定 GEMINI_API_KEY！")
    st.stop()

# 移除 transport="rest"，因為問題不在網路
genai.configure(api_key=api_key)
MODEL_NAME = 'gemini-3.8-flash'
STUDENT_ID = "user_kashiwa_01"

st.set_page_config(page_title="雙腦 AI 家教 (無狀態不死結版)", page_icon="🧠")
st.title("🧠 雙腦協作 AI 家教")

# ==========================================
# 1. SQLite 記憶庫
# ==========================================
DB_FILE = "tutor_memory.db"
def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute('CREATE TABLE IF NOT EXISTS mastery_profile (student_id TEXT, topic TEXT, last_mistake_note TEXT, PRIMARY KEY (student_id, topic))')
def get_last_mistake(student_id: str, topic: str) -> str:
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT last_mistake_note FROM mastery_profile WHERE student_id=? AND topic=?", (student_id, topic))
        result = cursor.fetchone()
        return result[0] if result else "無"
init_db()

# ==========================================
# 2. 左腦工具
# ==========================================
def calculate_sample_variance(numbers: list[float]) -> float:
    return float(np.var([float(x) for x in numbers], ddof=1))

def record_student_mistake(topic: str, mistake_note: str) -> str:
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute('INSERT INTO mastery_profile (student_id, topic, last_mistake_note) VALUES (?, ?, ?) ON CONFLICT DO UPDATE SET last_mistake_note=excluded.last_mistake_note', (STUDENT_ID, topic, mistake_note))
    return "已成功寫入。"

# ==========================================
# 3. 模型設定
# ==========================================
system_instruction = f"""
你是一位嚴格的統計學 AI 家教。
【學習紀錄】主題：樣本變異數 | 弱點：{get_last_mistake(STUDENT_ID, "樣本變異數")}
【守則】1. 遇到計算務必呼叫 `calculate_sample_variance`，但絕不直接給答案。2. 透過反問引導。3. 觀念錯時呼叫 `record_student_mistake`。
"""
def get_model():
    return genai.GenerativeModel(
        model_name=MODEL_NAME,
        tools=[calculate_sample_variance, record_student_mistake],
        system_instruction=system_instruction
    )

# ==========================================
# 4. Streamlit UI (徹底棄用 chat_session)
# ==========================================
# 自己維護乾淨的對話陣列，絕不依賴 SDK 狀態
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "model", "parts": ["你好！請給我一組數據，我們來算變異數。"]}]

for msg in st.session_state.messages:
    role = "assistant" if msg["role"] == "model" else "user"
    st.chat_message(role).markdown(msg["parts"][0])

if prompt := st.chat_input("輸入數據或回覆..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "parts": [prompt]})

    with st.chat_message("assistant"):
        with st.spinner("左腦計算與思考中..."):
            try:
                model = get_model()
                # 每次發送都是獨立請求 (Stateless)，把整個歷史紀錄餵進去
                response = model.generate_content(st.session_state.messages)
                
                # 攔截並手動處理工具呼叫
                fc = None
                if response.parts:
                    for part in response.parts:
                        if hasattr(part, 'function_call') and part.function_call.name:
                            fc = part.function_call
                            break
                            
                if fc:
                    func_name = fc.name
                    args = type(fc.args).to_dict(fc.args) if hasattr(type(fc.args), 'to_dict') else dict(fc.args)
                    
                    if func_name == "calculate_sample_variance":
                        res = calculate_sample_variance(**args)
                    elif func_name == "record_student_mistake":
                        res = record_student_mistake(**args)
                    else:
                        res = "未知"
                        
                    # 【核心解法】將計算結果偽裝成「系統訊息」暫時加入對話，直接向 AI 要最終回覆
                    temp_messages = st.session_state.messages.copy()
                    temp_messages.append({
                        "role": "user", 
                        "parts": [f"[系統自動運算結果] 左腦執行 {func_name} 的結果為：{res}。請以此結果為依據，用家教口吻繼續反問引導學生（絕不要告訴學生你呼叫了工具或看到了系統訊息）。"]
                    })
                    response = model.generate_content(temp_messages)

                # 提取最終文字並打字機輸出
                final_text = response.text if hasattr(response, 'text') else "（完成運算）"
                
                placeholder = st.empty()
                def fake_stream():
                    for char in final_text:
                        yield char
                        time.sleep(0.005)
                full_response = placeholder.write_stream(fake_stream)
                
                # 只把純文字存入歷史紀錄，保持歷史陣列絕對乾淨
                st.session_state.messages.append({"role": "model", "parts": [full_response]})
                
            except Exception as e:
                st.error(f"發生錯誤: {str(e)}")