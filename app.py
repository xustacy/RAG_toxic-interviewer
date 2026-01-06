import streamlit as st
import os
import gdown
import zipfile
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# ==========================================
# 1. 設定區：請填入您的 Google Drive File ID
# ==========================================
# 範例：如果連結是 https://drive.google.com/file/d/1xxxx/view...
# 這裡就填入 "1xxxx"
GDRIVE_FILE_ID = "1iwvWuIZlLRzirPlOZAwJhNlnCza9y5Yt" 

# ==========================================
# 2. 定義 Embedding 模型 (必須與建立時一致)
# ==========================================
class EmbeddingGemmaEmbeddings(HuggingFaceEmbeddings):
    def __init__(self, **kwargs):
        super().__init__(
            model_name="google/embeddinggemma-300m",
            encode_kwargs={"normalize_embeddings": True},
            **kwargs
        )

    def embed_documents(self, texts):
        # 修正：改成通用的標題，避免誤導
        texts = [f"title: 保險商品條款 | text: {t}" for t in texts]
        return super().embed_documents(texts)

    def embed_query(self, text):
        return super().embed_query(f"task: search result | query: {text}")

# ==========================================
# 3. 系統初始化與資料庫下載
# ==========================================
st.set_page_config(page_title="專業保險諮詢 AI", layout="wide")
st.title("🛡️ 專業保險諮詢與推薦系統")

if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    api_key = st.secrets["GROQ_API_KEY"]
else:
    st.error("❌ 未設定 GROQ_API_KEY，請至 Streamlit Secrets 進行設定。")
    st.stop()

@st.cache_resource
def load_resources():
    folder_name = "faiss_db_checkpoint"
    zip_name = "faiss_db_checkpoint.zip"
    
    # 1. 檢查資料庫是否存在，不存在則下載
    if not os.path.exists(folder_name):
        if not os.path.exists(zip_name):
            if "請將您的" in GDRIVE_FILE_ID:
                st.error("⚠️ 請先在 app.py 第 16 行填入正確的 Google Drive File ID！")
                st.stop()
                
            with st.spinner("📦 正在從雲端下載資料庫 (初次啟動需時較長)..."):
                try:
                    url = f'https://drive.google.com/uc?id={GDRIVE_FILE_ID}'
                    gdown.download(url, zip_name, quiet=False)
                except Exception as e:
                    st.error(f"下載失敗，請確認 File ID 正確且權限已開。錯誤: {e}")
                    st.stop()
        
        # 2. 解壓縮
        with st.spinner("📂 正在解壓縮資料庫..."):
            try:
                with zipfile.ZipFile(zip_name, 'r') as zip_ref:
                    zip_ref.extractall(".") # 解壓到當前目錄
            except Exception as e:
                st.error(f"解壓縮失敗: {e}")
                st.stop()

    # 3. 載入 FAISS
    try:
        embeddings = EmbeddingGemmaEmbeddings()
        db = FAISS.load_local(
            folder_name, 
            embeddings,
            allow_dangerous_deserialization=True
        )
        return db
    except Exception as e:
        st.error(f"資料庫讀取失敗：{e}")
        return None

vectorstore = load_resources()

if not vectorstore:
    st.stop()

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# 設定 LLM
llm = ChatOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=api_key,
    model="llama3-70b-8192", 
    temperature=0.3,         
)

# ==========================================
# 4. Prompt 設定
# ==========================================
persona_instruction = """
你是專業且充滿熱忱的保險業務員，致力於提供最優質的服務。
你擁有市面上幾家大型保險公司的所有保險商品資料。

請務必嚴格遵守以下規則：
1. **只能**根據下方的【已知資訊】來回答問題。
2. 若資料不足或題目超過能力範圍，請回答：「不好意思，目前的內部資料庫中沒有相關資訊，建議您直接洽詢該保險公司的專人客服服務。」
3. **拒絕回答**任何跟保險以外相關內容（例如：食譜、程式碼、旅遊景點等）。
4. 語氣保持親切友善、專業簡潔，並使用台灣繁體中文。
"""

qa_prompt = PromptTemplate(
    template=persona_instruction + """
    
    【已知資訊】：
    {context}
    
    使用者問題：{question}
    
    專業業務員回覆：
    """,
    input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": qa_prompt}
)

# ==========================================
# 5. 介面功能
# ==========================================
tab1, tab2 = st.tabs(["💬 線上保險諮詢", "📋 智能保險推薦"])

with tab1:
    st.subheader("有什麼保險問題我可以幫您嗎？")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("請輸入您的問題..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("正在查閱保險條款..."):
                try:
                    response = qa_chain.invoke({"query": prompt})
                    st.markdown(response["result"])
                    st.session_state.messages.append({"role": "assistant", "content": response["result"]})
                except Exception as e:
                    st.error(f"錯誤：{e}")

with tab2:
    st.subheader("為您量身打造的保險規劃")
    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("性別", ["男", "女"])
            age = st.number_input("年齡", 25, 100, 30)
            job = st.text_input("職業", "工程師")
        with col2:
            salary = st.selectbox("年收", ["50萬以下", "50-100萬", "100-200萬", "200萬以上"])
            budget = st.text_input("預算", "月繳 3000")
        
        ins_type = st.selectbox("險種", ["醫療險", "意外險", "儲蓄險", "旅遊險", "長照險", "壽險"])
        
        extra_info = ""
        if ins_type == "旅遊險":
            dest = st.text_input("國家")
            days = st.number_input("天數", 1, 365, 5)
            extra_info = f"去{dest}旅遊{days}天"

        if st.button("開始分析"):
            with st.spinner("分析中..."):
                query = f"使用者：{gender}, {age}歲, 職業{job}, 年收{salary}, 預算{budget}。想找{ins_type}。{extra_info}。請推薦商品。"
                response = qa_chain.invoke({"query": query})
                st.markdown(response["result"])