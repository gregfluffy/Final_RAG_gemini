# 這段程式碼示範如何連接到 Google Gemini API，並列出你可以使用的模型。
# 注意：請確保你已經安裝了 google-generativeai 套件，並且有一個有效的 API Key。
# 安裝套件的指令：
# pip install google-generativeai   
# 這段程式碼會嘗試從環境變數讀取 API Key，如果找不到，會提示你輸入。然後它會列出所有支援內容生成的模型。

import os
import google.generativeai as genai

# 建議作法：從環境變數讀取 API Key
# 如果你還沒設定環境變數，可以先暫時用下一行（把註解拿掉並填入你的 Key）：
# GOOGLE_API_KEY = "你的_API_KEY"
GOOGLE_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GOOGLE_API_KEY:
    # 備用方案：如果找不到環境變數，直接在這裡填入 Key（請注意隱私）
    GOOGLE_API_KEY = input("請輸入你的 API Key: ").strip()

# 設定你的 Key
genai.configure(api_key=GOOGLE_API_KEY)

# 列出所有你可以使用的模型
print("你的 API Key 可以使用的模型清單：")
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"名稱: {m.name}")
except Exception as e:
    print(f"連線或驗證失敗，錯誤訊息: {e}")
    print("請檢查你的網路連線，或是 API Key 是否正確且擁有權限。")