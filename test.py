import google.generativeai as genai
import os

api_key = os.environ.get("GEMINI_API_KEY", "").strip()
# 測試 1：先不加 transport="rest" 看看 gRPC 更新後是否已經正常
genai.configure(api_key=api_key)

model = genai.GenerativeModel('gemini-1.5-flash')
print("正在發送測試請求至 Gemini...")
try:
    response = model.generate_content("哈囉，請回覆我收到", request_options={"timeout": 30})
    print("API 連線成功！Gemini 回覆：", response.text)
except Exception as e:
    print("連線失敗：", e)
