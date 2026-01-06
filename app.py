import streamlit as st
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# ==========================================
# 1. 定義客製化 Embedding 類別 (關鍵修復！)
# ==========================================
# 這是為了配合您在 Colab 建立資料庫時使用的設定
class EmbeddingGemmaEmbeddings(HuggingFaceEmbeddings):
    def __init__(self, **kwargs):
        super().__init__(
            model_name="google/embeddinggemma-300m",
            encode_kwargs={"normalize_embeddings": True},
            **kwargs
        )

    def embed_documents(self, texts):
        # 修改這裡：改成通用的描述，或者直接用 "none"
        # 這樣就不會誤導模型以為所有資料都是南山的
        texts = [f"title: 保險商品條款 | text: {t}" for t in texts]
        return super().embed_documents(texts)

    def embed_query(self, text):
        # 這是問答時最關鍵的一行，保持原樣即可
        return super().embed_query(f"task: search result | query: {text}")
# ==========================================
# 2. 系統設定與初始化
# ==========================================
st.set_page_config(page_title="專業保險諮詢 AI", layout="wide")
st.title("🛡️ 專業保險諮詢與推薦系統")

# 檢查 API Key
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    api_key = st.secrets["GROQ_API_KEY"]
else:
    st.error("❌ 未設定 GROQ_API_KEY，請至 Streamlit Secrets 進行設定。")
    st.stop()

# 載入資料庫 (使用快取)
@st.cache_resource
def load_resources():
    try:
        # 使用剛剛定義的 Gemma 模型載入
        embeddings = EmbeddingGemmaEmbeddings()
        
        # 載入 FAISS 資料庫
        # 注意：根據您的 GitHub 截圖，資料夾名稱是 'faiss_db_checkpoint'
        db = FAISS.load_local(
            "faiss_db_checkpoint", 
            embeddings,
            allow_dangerous_deserialization=True
        )
        return db
    except Exception as e:
        # 把具體錯誤 print 出來，方便除錯
        print(f"詳細錯誤訊息: {e}")
        return None

# 初始化資源
vectorstore = load_resources()

if not vectorstore:
    st.error("⚠️ 資料庫載入失敗！請確認：\n1. GitHub 上是否有 'faiss_db_checkpoint' 資料夾？\n2. 該資料夾內是否有 'index.faiss' 和 'index.pkl'？\n3. 是否使用了 Git LFS 上傳大檔案？")
    # 這裡可以顯示更多除錯資訊
    st.info("提示：如果這是第一次部署，請確保 index.faiss 檔案大小正常 (不是 1KB 的指針檔)。")
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
# 3. 定義 Prompt Templates
# ==========================================

persona_instruction = """
你是專業且充滿熱忱的保險業務員，致力於提供最優質的服務。
你擁有市面上幾家大型保險公司的所有保險商品資料。

請務必嚴格遵守以下規則：
1. **只能**根據下方的【已知資訊】來回答問題。
2. 若資料不足或題目超過能力範圍（例如資料庫沒有該商品），請回答：「不好意思，目前的內部資料庫中沒有相關資訊，建議您直接洽詢該保險公司的專人客服服務。」
3. **拒絕回答**任何跟保險以外相關內容（例如：食譜、程式碼、旅遊景點介紹、巴斯克蛋糕怎麼做等），請禮貌拒絕並將話題引導回保險。
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

# 建立檢索問答鏈
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": qa_prompt}
)

# ==========================================
# 4. 介面功能實作
# ==========================================

tab1, tab2 = st.tabs(["💬 線上保險諮詢", "📋 智能保險推薦"])

# --- 功能一：Chatbot ---
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
                    result = response["result"]
                    st.markdown(result)
                    st.session_state.messages.append({"role": "assistant", "content": result})
                except Exception as e:
                    st.error(f"發生錯誤，請稍後再試：{e}")

# --- 功能二：保險推薦 ---
with tab2:
    st.subheader("為您量身打造的保險規劃")
    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("性別", ["男", "女"])
            age = st.number_input("年齡", min_value=0, max_value=100, value=30)
            job = st.text_input("職業", "一般內勤")
        with col2:
            salary = st.selectbox("年收入範圍", ["50萬以下", "50-100萬", "100-200萬", "200萬以上"])
            budget = st.text_input("預算 (月繳/年繳)", "月繳 3000 元")
        
        ins_type = st.selectbox(
            "您感興趣的保險類型", 
            ["醫療險", "意外險", "儲蓄險/投資型", "旅遊平安險", "長照險", "壽險"]
        )
        
        travel_details = ""
        if ins_type == "旅遊平安險":
            st.info("✈️ 偵測到旅遊需求，請補充細節：")
            c1, c2 = st.columns(2)
            with c1:
                dest = st.text_input("旅遊國家", "日本")
            with c2:
                days = st.number_input("旅遊天數", min_value=1, value=5)
            travel_details = f"，旅遊目的地為{dest}，預計旅遊{days}天"

        if st.button("🚀 開始分析並推薦", type="primary"):
            with st.spinner("正在分析您的需求並比對資料庫..."):
                user_profile_query = f"""
                使用者基本資料：
                - 性別：{gender}
                - 年齡：{age}
                - 職業：{job}
                - 年收入：{salary}
                - 預算：{budget}
                - 主要需求：{ins_type}{travel_details}
                
                任務：請推薦適合的【{ins_type}】商品並說明原因。
                """
                try:
                    response = qa_chain.invoke({"query": user_profile_query})
                    st.success("分析完成！以下是給您的專業建議：")
                    st.markdown(response["result"])
                except Exception as e:
                    st.error(f"分析失敗：{e}")