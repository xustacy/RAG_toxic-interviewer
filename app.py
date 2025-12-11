import streamlit as st
import os
import pypdf
import aisuite as ai
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import login

# --- 1. 設定頁面 ---
st.set_page_config(page_title="毒舌面試官", page_icon="☠️")

# --- 2. 處理 API Keys (從 Streamlit Secrets 讀取) ---
# 注意：在本地測試時若報錯，請確保你有設定 secrets.toml，或暫時用環境變數
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    HF_TOKEN = st.secrets["HF_TOKEN"]
    
    os.environ['GROQ_API_KEY'] = GROQ_API_KEY
    login(token=HF_TOKEN)
except Exception as e:
    st.error("API Key 設定有誤，請檢查 Secrets 設定。")
    st.stop()

# --- 3. 載入 RAG 模型 (增加快取以加速) ---
@st.cache_resource
def load_vector_db():
    class EmbeddingGemmaEmbeddings(HuggingFaceEmbeddings):
        def __init__(self, **kwargs):
            super().__init__(
                model_name="google/embeddinggemma-300m",
                encode_kwargs={"normalize_embeddings": True},
                **kwargs
            )
        def embed_documents(self, texts):
            texts = [f"title: Job Description | text: {t}" for t in texts]
            return super().embed_documents(texts)
        def embed_query(self, text):
            return super().embed_query(f"task: search job description | query: {text}")

    embedding_model = EmbeddingGemmaEmbeddings()
    
    # 檢查資料庫是否存在
    if not os.path.exists("faiss_db"):
        st.error("找不到 faiss_db 資料夾，請確認已上傳至 GitHub。")
        return None

    vectorstore = FAISS.load_local(
        "faiss_db",
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )
    return vectorstore.as_retriever(search_kwargs={"k": 3})

retriever = load_vector_db()

# --- 4. 設定 LLM ---
client = ai.Client()
# 使用修正後的模型名稱
model = "groq:llama-3.3-70b-versatile" 

# --- 5. 核心邏輯函式 ---
def extract_text_from_pdf(pdf_file):
    try:
        reader = pypdf.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        return f"Error: {str(e)}"

def get_ai_response(job_title, resume_text):
    if not retriever:
        return "系統錯誤：資料庫未載入"

    # RAG 檢索
    search_query = f"{job_title} requirements skills"
    docs = retriever.invoke(search_query)
    retrieved_chunks = "\n\n".join([doc.page_content for doc in docs])

    # Prompt
    system_prompt = """
    你是一位在科技業界打滾20多年的資深面試官。
    個性設定：極度毒舌、講話帶刺、幽默但一針見血、沒有耐心。
    任務：根據「市場標準」對求職者的「履歷」進行無情吐槽，但最後必須給出基於 CoT 的具體建議。
    """
    
    user_prompt = f"""
    ### 目標職位：{job_title}
    ### 市場標準 JD：
    {retrieved_chunks}
    ### 求職者履歷：
    {resume_text}
    
    請開始你的毒舌評論與建議：
    """

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"發生錯誤: {str(e)}"

# --- 6. UI 介面 ---
st.title("☠️ 毒舌面試官模擬器")
st.markdown("上傳履歷，讓 AI 用市場標準狠狠地「指導」你。")

col1, col2 = st.columns(2)
with col1:
    job_input = st.text_input("你想應徵什麼職位？", "前端工程師")
with col2:
    uploaded_file = st.file_uploader("上傳履歷 (PDF)", type="pdf")

if st.button("開始面試", type="primary"):
    if not uploaded_file:
        st.warning("請先上傳履歷！")
    elif not job_input:
        st.warning("請輸入職位名稱！")
    else:
        with st.spinner("面試官正在翻白眼並閱讀你的履歷..."):
            resume_text = extract_text_from_pdf(uploaded_file)
            result = get_ai_response(job_input, resume_text)
            st.markdown("### 面試官回饋：")
            st.write(result)
