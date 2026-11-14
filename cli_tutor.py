
import os
import json
import sqlite3
import math
import requests

from dotenv import load_dotenv


# ==================================================
# 0. 環境設定
# ==================================================

# 本機開發時讀取 .env
# Codespaces 則優先使用 GitHub Secrets 環境變數
load_dotenv()

API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()

if not API_KEY:
    raise RuntimeError(
        "找不到 GEMINI_API_KEY，請檢查 Codespaces Secrets 或本機 .env"
    )

# 使用目前支援 Function Calling 的 Gemini Flash 模型
MODEL = "gemini-3.5-flash"

URL = (
    f"https://generativelanguage.googleapis.com/"
    f"v1beta/models/{MODEL}:generateContent"
)

HEADERS = {
    "Content-Type": "application/json",
    "x-goog-api-key": API_KEY
}

STUDENT_ID = "user_raw_01"
DB_FILE = "tutor_memory.db"

REQUEST_TIMEOUT = 45
MAX_TOOL_ROUNDS = 8


# ==================================================
# 1. SQLite 學習記憶庫
# ==================================================

def init_db():
    with sqlite3.connect(DB_FILE) as conn:

        conn.execute("""
            CREATE TABLE IF NOT EXISTS mastery_profile (
                student_id TEXT NOT NULL,
                topic TEXT NOT NULL,
                last_mistake_note TEXT,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (student_id, topic)
            )
        """)

        # 舊版資料庫相容處理：
        # 如果原本資料表沒有 updated_at，就自動新增
        columns = {
            row[1]
            for row in conn.execute(
                "PRAGMA table_info(mastery_profile)"
            )
        }

        if "updated_at" not in columns:
            conn.execute("""
                ALTER TABLE mastery_profile
                ADD COLUMN updated_at TEXT
            """)


def get_last_mistake(student_id, topic):

    with sqlite3.connect(DB_FILE) as conn:

        cursor = conn.execute("""
            SELECT last_mistake_note
            FROM mastery_profile
            WHERE student_id = ? AND topic = ?
        """, (student_id, topic))

        row = cursor.fetchone()

        return row[0] if row else "無"


def record_student_mistake(topic, mistake_note):

    if not topic or not mistake_note:
        raise ValueError("主題及錯誤紀錄不可為空")

    print(f"\n💾 [記憶寫入] {topic}: {mistake_note}")

    with sqlite3.connect(DB_FILE) as conn:

        conn.execute("""
            INSERT INTO mastery_profile
                (student_id, topic, last_mistake_note, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)

            ON CONFLICT(student_id, topic)
            DO UPDATE SET
                last_mistake_note = excluded.last_mistake_note,
                updated_at = CURRENT_TIMESTAMP
        """, (STUDENT_ID, topic, mistake_note))

    return {
        "status": "success",
        "message": "學習錯誤紀錄已儲存"
    }


init_db()


# ==================================================
# 2. Python 工具：數學計算
# ==================================================

def calculate_sample_variance(numbers):

    print(f"\n⚙️ [Python 計算引擎] 數據: {numbers}")

    if not isinstance(numbers, list):
        raise ValueError("輸入必須是數字列表")

    if len(numbers) < 2:
        raise ValueError("樣本變異數至少需要兩個數字")

    values = [float(x) for x in numbers]

    if not all(math.isfinite(x) for x in values):
        raise ValueError("數據不能包含 NaN 或無限大")

    variance = float(
        sum((x - sum(values) / len(values)) ** 2
            for x in values)
        / (len(values) - 1)
    )

    return {
        "status": "success",
        "sample_variance": variance,
        "sample_size": len(values)
    }


# ==================================================
# 3. Gemini Function Calling 工具定義
# ==================================================

tools_schema = [
    {
        "functionDeclarations": [

            {
                "name": "calculate_sample_variance",

                "description": (
                    "使用 Python 計算一組數字的樣本變異數。"
                    "當學生提供數據或詢問樣本變異數計算時必須使用。"
                ),

                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "numbers": {
                            "type": "ARRAY",
                            "items": {
                                "type": "NUMBER"
                            },
                            "description": "要計算的數字列表"
                        }
                    },
                    "required": ["numbers"]
                }
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
                            "type": "STRING"
                        },
                        "mistake_note": {
                            "type": "STRING"
                        }
                    },
                    "required": [
                        "topic",
                        "mistake_note"
                    ]
                }
            }

        ]
    }
]


# ==================================================
# 4. AI 家教系統提示
# ==================================================

topic = "樣本變異數"

past_mistake = get_last_mistake(
    STUDENT_ID,
    topic
)

system_prompt = f"""
你是一位嚴格、耐心、重視邏輯的統計學家教。

你的工作是教導學生理解統計學，而不是直接給答案。

【學生資料】
學生編號：{STUDENT_ID}

目前學習主題：{topic}

過去錯誤紀錄：
{past_mistake}

【教學原則】

1. 當學生提供一組數字並要求計算樣本變異數時，
   必須使用 calculate_sample_variance。
   不可以自行估算或直接編造計算結果。

2. 計算工作交給 Python。
   你負責解釋數學概念及計算結果。

3. 採用蘇格拉底式教學。
   透過反問，引導學生理解平均數、偏差、
   平方偏差及自由度。

4. 不要一次給出所有答案。
   優先讓學生自己思考。

5. 如果學生出現明確的統計觀念錯誤，
   必須使用 record_student_mistake 記錄錯誤。

6. 如果學生只是提問或尚未回答，
   不要錯誤地將其記錄為觀念錯誤。

7. 使用繁體中文回答。

8. Python 計算結果是數學答案的依據，
   不可修改或捏造計算結果。
"""


# ==================================================
# 5. 工具執行引擎
# ==================================================

def execute_tool(function_name, args):

    try:

        if function_name == "calculate_sample_variance":

            result = calculate_sample_variance(
                args.get("numbers", [])
            )

        elif function_name == "record_student_mistake":

            result = record_student_mistake(
                args.get("topic"),
                args.get("mistake_note")
            )

        else:

            return {
                "status": "error",
                "message": f"未知工具：{function_name}"
            }

        return result

    except Exception as e:

        return {
            "status": "error",
            "message": str(e)
        }


# ==================================================
# 6. Gemini API 呼叫
# ==================================================

def call_gemini(history):

    payload = {
        "systemInstruction": {
            "parts": [
                {"text": system_prompt}
            ]
        },

        "contents": history,

        "tools": tools_schema
    }

    response = requests.post(
        URL,
        headers=HEADERS,
        json=payload,
        timeout=REQUEST_TIMEOUT
    )

    if not response.ok:

        try:
            error_info = response.json().get("error", {})
            message = error_info.get("message", response.text)
            code = error_info.get("code", response.status_code)

        except ValueError:
            message = response.text
            code = response.status_code

        raise RuntimeError(
            f"Gemini API 錯誤 {code}: {message}"
        )

    data = response.json()

    candidates = data.get("candidates", [])

    if not candidates:
        raise RuntimeError(
            "Gemini 沒有回傳候選答案，請檢查安全過濾或 API 回應"
        )

    candidate = candidates[0]

    content = candidate.get("content")

    if not content:
        raise RuntimeError(
            f"Gemini 沒有回傳內容，finishReason="
            f"{candidate.get('finishReason')}"
        )

    return content


# ==================================================
# 7. AI Agent 對話及工具呼叫迴圈
# ==================================================

def process_user_message(user_input, history):

    # 加入學生訊息
    history.append({
        "role": "user",
        "parts": [
            {"text": user_input}
        ]
    })

    for round_number in range(MAX_TOOL_ROUNDS):

        print("\n⏳ Gemini AI 思考中...")

        # 向 Gemini 發出請求
        model_content = call_gemini(history)

        parts = model_content.get("parts", [])

        # 找出所有 Function Calls
        function_calls = [
            part["functionCall"]
            for part in parts
            if "functionCall" in part
        ]

        # 將模型原始回應完整加入歷史
        history.append(model_content)

        # 如果沒有工具呼叫，表示 AI 已完成回答
        if not function_calls:

            final_text = "\n".join(
                part["text"]
                for part in parts
                if "text" in part
            )

            if not final_text:
                final_text = "（AI 沒有產生文字回覆）"

            return final_text

        # 執行所有工具呼叫
        function_responses = []

        for fc in function_calls:

            function_name = fc.get("name")
            args = fc.get("args", {})

            print(
                f"\n🔧 [Function Calling] "
                f"{function_name}"
            )

            print(
                f"   參數：{json.dumps(args, ensure_ascii=False)}"
            )

            result = execute_tool(
                function_name,
                args
            )

            print(
                f"   結果：{json.dumps(result, ensure_ascii=False)}"
            )

            response_part = {
                "functionResponse": {
                    "name": function_name,
                    "response": {
                        "result": result
                    }
                }
            }

            # 如果模型提供 Function Call ID，
            # 必須在對應的 Function Response 中回傳
            if fc.get("id"):
                response_part["functionResponse"]["id"] = fc["id"]

            function_responses.append(response_part)

        # 將所有工具執行結果一次傳回 Gemini
        history.append({
            "role": "user",
            "parts": function_responses
        })

    raise RuntimeError(
        "超過最大工具呼叫次數，請重新提出問題"
    )


# ==================================================
# 8. 主程式
# ==================================================

def main():

    history = []

    print("=" * 55)
    print("🧠 Stat-Tutor AI 統計家教啟動")
    print("=" * 55)

    print(f"模型：{MODEL}")
    print(f"學生編號：{STUDENT_ID}")
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

            final_text = process_user_message(
                user_input,
                history
            )

            print("\n🤖 家教：")
            print(final_text)

        except KeyboardInterrupt:

            print("\n\n👋 使用者中斷程式。")
            break

        except requests.exceptions.Timeout:

            print(
                "\n❌ API 連線逾時。"
                "請檢查網路或稍後重試。"
            )

        except requests.exceptions.ConnectionError:

            print(
                "\n❌ 無法連接 Gemini API。"
                "請檢查 Codespaces 網路。"
            )

        except requests.exceptions.RequestException as e:

            print(f"\n❌ HTTP 請求錯誤：{e}")

        except Exception as e:

            print(f"\n❌ 系統錯誤：{e}")


if __name__ == "__main__":
    main()
