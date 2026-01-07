# 🛡️ 專業保險諮詢與推薦系統 (Insurance RAG Agent)

## 🚩 部署網址：https://raginsurancebroker-2qvj3p9ccqxwksambxxvyp.streamlit.app/
這是 **RAG** 技術的智能保險諮詢系統。它結合了 **Streamlit** 的互動介面、 **FAISS** 向量資料庫以及使用 **Groq (LLM)**，能夠根據內部的保險條款文件，提供精確的問答與商品推薦。

## ✨ 主要功能

1.  **💬 線上保險諮詢 (AI Chatbot)**
    * 使用者可用自然語言提問（例如：「理賠需要哪些文件？」）。
    * 系統會透過 RAG 技術檢索相關條款，避免 AI 產生幻覺 (Hallucination)。
    * 具備「工程師模式 (Debug Mode)」，可查看 AI 實際參考了哪些文件來源。

2.  **📋 智能保險推薦**
    * 透過表單輸入個人條件（年齡、職業、預算、需求）。
    * AI 綜合分析後，推薦適合的險種與商品，並說明推薦理由。
    * 支援情境式推薦（例如：日本旅遊險、意外險規劃）。

3.  **🚀 自動化資料庫部署**
    * 由於Faiss資料庫容量太大，因此直接使用雲端，系統啟動時自動從 Google Drive 下載最新的向量資料庫 (`faiss_db_mini.zip`)。

## 🛠️ 技術架構

* **Frontend**: [Streamlit](https://streamlit.io/)
* **LLM (推論引擎)**: [Groq API](https://groq.com/) (Model: `llama-3.3-70b-versatile` / `llama-3.1-8b-instant`)
* **Orchestration**: [LangChain](https://www.langchain.com/)
* **Vector Database**: [FAISS](https://github.com/facebookresearch/faiss) (CPU version)
* **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace)
* **Deployment**: Streamlit Cloud
