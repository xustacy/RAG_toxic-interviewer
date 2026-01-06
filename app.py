import streamlit as st
import os
import gdown
import zipfile
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

# ==========================================
# 1. 系統設定
# ==========================================
st.set_page_config(page_title="專業保險諮詢 AI", layout="wide")
st.title("🛡️ 專業保險諮詢與推薦系統")

# 檢查 Groq 金鑰
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    api_key = st.secrets["GROQ_API_KEY"]
else:
    st.error("❌ 未設定 GROQ_API_KEY，請至 Streamlit Secrets 設定。")
    st.stop()

# ==========================================
# 2. 設定 Google Drive 下載
# ==========================================
# 這是您剛剛提供的檔案 ID
GDRIVE_FILE_ID = "1SWLCi36AvdoOO8oTAflVD9luHyDKQbRL" 
ZIP_NAME = "faiss_db_mini.zip"
DB_FOLDER = "faiss_db_mini"

# ==========================================
# 3. 定義 Embedding 模型 (關鍵修改！)
# ==========================================
def get_embeddings():
    """使用與資料庫一致的 MiniLM 模型 (維度 384)"""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

# ==========================================
# 4. 載入資源 (下載 -> 解壓 -> 讀取)
# ==========================================
@st.cache_resource
def load_resources():
    # 下載與解壓縮
    if not os.path.exists(DB_FOLDER):
        if not os.path.exists(ZIP_NAME):
            with st.spinner("📦 正在從雲端下載資料庫..."):
                try:
                    url = f'https://drive.google.com/uc?id={GDRIVE_FILE_ID}'
                    gdown.download(url, ZIP_NAME, quiet=False)
                except Exception as e:
                    st.error(f"下載失敗: {e}")
                    return None
        
        with st.spinner("📂 解壓縮資料庫..."):
            try:
                with zipfile.ZipFile(ZIP_NAME, 'r') as zip_ref:
                    zip_ref.extractall(".")
            except Exception as e:
                st.error(f"解壓縮失敗: {e}")
                return None

    # 載入 FAISS
    try:
        embeddings = get_embeddings()
        
        # 嘗試載入資料庫
        # 優先檢查 faiss_db_mini 資料夾，若無則檢查當前目錄
        if os.path.exists(DB_FOLDER):
            load_path = DB_FOLDER
        else:
            load_path = "."
            
        db = FAISS.load_local(
            load_path, 
            embeddings,
            allow_dangerous_deserialization=True
        )
        st.success("✅ 資料庫載入成功！")
        return db
    except Exception as e:
        st.error(f"資料庫讀取失敗：{e}")
        st.info("提示：請確認 requirements.txt 是否包含 faiss-cpu 與 sentence-transformers")
        return None

vectorstore = load_resources()

if not vectorstore:
    st.stop()

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# 設定 LLM (使用 Groq)
llm = ChatOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=api_key,
    model="llama3-70b-8192", 
    temperature=0.3,         
)

# ==========================================
# 5. Prompt 與 Chain 設定
# ==========================================
persona_instruction = """
你是專業且充滿熱忱的保險業務員，致力於提供最優質的服務。
請務必嚴格遵守以下規則：
1. **只能**根據下方的【已知資訊】來回答問題。
2. 若資料不足或題目超過能力範圍，請回答：「不好意思，目前的內部資料庫中沒有相關資訊，建議您直接洽詢該保險公司的專人客服服務。」
3. **拒絕回答**任何跟保險以外相關內容。
4. 語氣保持親切友善、專業簡潔，並使用台灣繁體中文。
"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", persona_instruction + "\n\n【已知資訊】：\n{context}"),
    ("human", "{question}")
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

qa_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | qa_prompt
    | llm
    | StrOutputParser()
)

# ==========================================
# 6. 介面功能
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
                    response = qa_chain.invoke(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
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
                response = qa_chain.invoke(query)
                st.markdown(response)