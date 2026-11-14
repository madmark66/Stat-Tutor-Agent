import os
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv

# 1. 載入隱藏的金鑰
load_dotenv()
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# 2. 定義 Python 數學左腦 (Tool)
# 必須寫清楚 docstring，Gemini 才知道什麼時候該呼叫它
def calculate_variance(data_list: list[float]) -> dict:
    """
    計算給定數據的樣本變異數與不偏變異數 (統計檢定常用)
    """
    data = np.array(data_list)
    return {
        "sample_variance": float(np.var(data)),
        "unbiased_variance": float(np.var(data, ddof=1))
    }

# 3. 綁定 Gemini 右腦與工具
# 3. 綁定 Gemini 右腦與工具
# 加上 system_instruction 來賦予它「家教」的人設與嚴格規則
socratic_instruction = """
你是一位準備日本統計檢定準一級的嚴格但友善的家教。
你的任務是引導學生思考，而不是直接給答案。

規則 1：當學生詢問數據的計算結果或提交他的計算答案時，你必須在背景呼叫工具取得正確答案。
規則 2：【絕對不可以】直接把數字答案告訴學生。
規則 3：根據你呼叫工具得到的正確答案，比對學生的答案。如果不一樣，請反問他計算步驟（例如：「你是除以 n 還是 n-1 呢？」）；如果他算對了，請給予鼓勵。
規則 4：請全程使用自然、專業的日文與學生互動。
"""

model = genai.GenerativeModel(
    model_name='gemini-3.8-flash',
    tools=[calculate_variance],
    system_instruction=socratic_instruction
)

chat = model.start_chat(enable_automatic_function_calling=True)

print("🤖 系統提示：正在與蘇格拉底家教對話...")
# 我們把 prompt 改成「學生算錯」的情境
prompt = "我有一組數據：12.5, 14.2, 13.8, 15.1, 14.0。我算出來的標本分散是 0.8770，對嗎？"

try:
    response = chat.send_message(prompt)
    print("✅ 家教回覆：")
    print(response.text)
except Exception as e:
    print(f"\n❌ 發生錯誤：{e}")

# 4. 啟動對話並開啟自動呼叫工具功能
chat = model.start_chat(enable_automatic_function_calling=True)

print("🤖 系統提示：正在將數學題目發送給 Gemini...")
# 故意用自然語言丟給它，測試它會不會自己抓出數字去呼叫 Python
prompt = "我有一組產線的感測器數據：12.5, 14.2, 13.8, 15.1, 14.0。請幫我算出這組數據的樣本變異數與不偏變異數，並用日文簡單回覆結果。"

response = chat.send_message(prompt)
print("\n✅ Gemini 回覆：")
print(response.text)