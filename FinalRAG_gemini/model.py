"""
# model.py
# 這是一個使用 Flask 框架構建的簡單 Web 應用程式，允許使用者上傳 PDF 文件，從中提取文字，
# 並使用 RAG（Retrieval-Augmented Generation）技術結合 Google Gemini API 來回答使用者的問題。
# 主要功能包括：
# 1. PDF 上傳與文字提取：使用 pypdf 庫從 PDF 文件中提取文字，並將其切分成小塊（chunks）以便後續處理。
# 2. RAG 切塊索引：將提取的文字切分成互相重疊的小方塊，確保語意不被切斷，並儲存在全域變數中。
# 3. 雙語 RAG 優化：當使用者提出問題時，檢測問題是否包含中文，如果是，則使用 Gemini API 將問題轉化為適合在英文教科書中搜尋的英文關鍵字，以提升 TF-IDF 的搜尋精準度。
# 4. 本地 RAG 檢索：使用 TF-IDF 向量化技術計算使用者問題與切塊文本的相似度，並撈取相似度最高的前 5 個文本塊作為背景知識。
# 5. Gemini API 呼叫：將相關的英文文本塊和使用者的問題一起送給 Gemini 2.5 Flash 模型，生成最終的回答，並強制要求 AI 使用繁體中文回答。
# 6. 錯誤處理：對於 PDF 解析失敗、問題為空、未上傳文件等情況進行適當的錯誤處理和回應。
# 注意：在運行此應用程式之前，請確保已安裝所需的 Python 庫（Flask、google-generativeai、pypdf、numpy、scikit-learn）並設定好 Gemini API Key。
# 這個應用程式的目的是展示如何結合本地文本處理和雲端 AI 模型來實現一個智能的文件問答系統，特別適合用於處理大量文本資料並從中提取有用信息來回答使用者的問題。
# 你可以將此代碼保存為 model.py，並確保在同一目錄下有一個 templates 文件夾，裡面包含 interface.html 這個前端界面文件，以便用戶能夠上傳 PDF 和提出問題。
# 運行此 Flask 應用程式後，使用者可以通過瀏覽器訪問 http://localhost:5000 來使用這個智能文件問答系統。
# 安裝套件：
# pip install Flask google-generativeai pypdf numpy scikit-learn
"""

import os
from flask import Flask, request, jsonify, render_template
import google.generativeai as genai
from pypdf import PdfReader
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# 設定你的 Gemini API Key
GOOGLE_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GOOGLE_API_KEY:
    GOOGLE_API_KEY = input("請輸入你的 Gemini API Key: ").strip()

genai.configure(api_key=GOOGLE_API_KEY)

# 全域變數
pdf_chunks = []  # 儲存切塊後的文字片段群組
total_extracted_chars = 0  # 記錄原本的總字數

def extract_text_from_pdf(pdf_path):
    """讀取 PDF 檔案並將所有頁面的文字合併"""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        print(f"PDF 解析失敗: {e}")
        return ""

def make_chunks(text, chunk_size=800, overlap=150):
    """【RAG 核心 1】將長文本切成互相重疊的小方塊 (Chunks)"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)  # 讓前後方塊有重疊，確保語意不被切斷
    return chunks

@app.route('/')
def index():
    return render_template('interface.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """處理 PDF 上傳並進行本地端 RAG 切塊索引"""
    global pdf_chunks, total_extracted_chars
    if 'pdfFile' not in request.files:
        return jsonify({"success": False, "message": "沒有找到檔案區塊"}), 400
    
    file = request.files['pdfFile']
    if file.filename == '':
        return jsonify({"success": False, "message": "未選取檔案"}), 400

    if file and file.filename.endswith('.pdf'):
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)
        file.save(file_path)

        # 1. 提取完整文字
        full_text = extract_text_from_pdf(file_path)
        
        if os.path.exists(file_path):
            os.remove(file_path)

        if not full_text.strip():
            return jsonify({"success": False, "message": "無法從該 PDF 中提取文字內容"}), 400

        total_extracted_chars = len(full_text)

        # 2. 執行切塊：把整本大書切成數百個 800 字的小方塊
        pdf_chunks = make_chunks(full_text, chunk_size=800, overlap=150)
        print(f"✅ 成功將文件切分為 {len(pdf_chunks)} 個獨立知識區塊。")

        return jsonify({
            "success": True, 
            "message": f"成功讀取文件！共提取了 {total_extracted_chars} 個字元。"
        })
    
    return jsonify({"success": False, "message": "檔案格式不正確，僅接受 PDF"}), 400

@app.route('/ask', methods=['POST'])
def ask_question():
    """接收提問，加入雙語 RAG 語意優化，結合 PDF 內容呼叫 Gemini API"""
    global pdf_chunks
    data = request.json
    user_query = data.get("question", "").strip()

    if not user_query:
        return jsonify({"success": False, "message": "問題不能為空"}), 400
    
    if not pdf_chunks:
        return jsonify({"success": False, "message": "請先上傳並分析 PDF 文件"}), 400

    try:
        model = genai.GenerativeModel('gemini-2.5-flash')

        # 💡【雙語 RAG 核心優化】
        # 如果使用者輸入中文，但我們的書是英文，直接比對會失敗。
        # 這裡利用 Gemini 速度極快的優勢，在背景將問題轉化為英文關鍵字來提升 TF-IDF 的搜尋精準度。
        search_query = user_query
        # 簡單偵測問題中是否包含中文
        if any('\u4e00' <= char <= '\u9fff' for char in user_query):
            translation_prompt = f"請將以下的使用者提問，轉化為適合在英文教科書中搜尋的簡短英文關鍵字或核心問句（只需要輸出英文，不要有任何額外解釋）：\n{user_query}"
            try:
                translated_response = model.generate_content(translation_prompt)
                search_query = translated_response.text.strip()
                print(f"🌐 雙語優化：將中文問題轉化為英文關鍵字進行本地檢索 -> '{search_query}'")
            except Exception as trans_err:
                print(f"翻譯輔助失敗，降級使用原句檢索: {trans_err}")

        # 1. 使用轉化後的 search_query 來做向量比對
        all_texts = pdf_chunks + [search_query]
        vectorizer = TfidfVectorizer(stop_words='english') # 加入 stop_words 忽略英文贅字（如 the, is, at）
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        
        # 2. 計算餘弦相似度
        query_vector = tfidf_matrix[-1]
        chunks_vectors = tfidf_matrix[:-1]
        similarities = cosine_similarity(query_vector, chunks_vectors).flatten()
        
        # 3. 提高容錯率：撈出相似度最高的前 5 個文本塊 (Top-5 Chunks)
        top_k_indices = np.argsort(similarities)[-5:][::-1]
        
        # 4. 拼裝背景知識
        relevant_context = ""
        matched_count = 0
        for idx in top_k_indices:
            if similarities[idx] > 0:
                relevant_context += f"--- 文件片段 {matched_count+1} ---\n{pdf_chunks[idx]}\n\n"
                matched_count += 1

        print(f"🔍 本地 RAG 成功匹配了 {matched_count} 個英文相關段落送給 Gemini。")

        # 5. 送給 Gemini 2.5 Flash 生成最終回答（強迫 AI 用繁體中文回答）
        prompt = f"""
你是一個專門精準導讀文件的 AI 助理。請仔細閱讀以下由系統自動為你從幾百頁大書中檢索出來的英文《精選相關文件段落》，並根據這些段落，使用【繁體中文】精準且詳細地回答使用者的《問題》。

《精選相關文件段落（英文）》：
{relevant_context}

《問題（使用者想了解的內容）》：
{user_query}

請注意：
1. 請完全使用【繁體中文】來回答問題，但專有名詞（如 CPU, Pipeline, Cache）可保留英文。
2. 如果上述提供的英文文件片段裡完全沒有跟問題相關的資訊，請禮貌地回答「根據您上傳的文件，找不到直接相關的解答」。
"""
        response = model.generate_content(prompt)
        return jsonify({
            "success": True,
            "answer": response.text
        })

    except Exception as e:
        return jsonify({"success": False, "message": f"RAG 檢索或 API 呼叫失敗: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)