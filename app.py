import streamlit as st
import os
import gdown
import zipfile
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq 

# ==========================================
# 1. 系統設定
# ==========================================
st.set_page_config(page_title="專業保險諮詢 AI", layout="wide")
st.title("🛡️ 專業保險諮詢與推薦系統 (V3.0 智能版)")

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
GDRIVE_FILE_ID = "1SWLCi36AvdoOO8oTAflVD9luHyDKQbRL" 
ZIP_NAME = "faiss_db_mini.zip"
DB_FOLDER = "faiss_db_mini"

# ==========================================
# 3. Embedding 模型
# ==========================================
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

# ==========================================
# 4. 載入資源 (⚠️ 修正點：純淨版，不含任何 UI 指令)
# ==========================================
@st.cache_resource(show_spinner=False) # 關閉內建 spinner，完全由我們控制
def load_resources():
    """
    這個函式只負責運算與資料讀取，
    絕對不包含 st.spinner, st.error 等 UI 互動。
    """
    # 1. 下載與解壓縮 (只做動作，不顯示 st 訊息)
    if not os.path.exists(DB_FOLDER):
        if not os.path.exists(ZIP_NAME):
            try:
                url = f'https://drive.google.com/uc?id={GDRIVE_FILE_ID}'
                gdown.download(url, ZIP_NAME, quiet=False)
            except:
                return None # 失敗就回傳 None，讓外面處理
        
        try:
            with zipfile.ZipFile(ZIP_NAME, 'r') as zip_ref:
                zip_ref.extractall(".")
        except:
            return None

    # 2. 載入 FAISS
    try:
        embeddings = get_embeddings()
        if os.path.exists(DB_FOLDER):
            load_path = DB_FOLDER
        else:
            load_path = "."
            
        db = FAISS.load_local(
            load_path, 
            embeddings,
            allow_dangerous_deserialization=True
        )
        return db
    except:
        return None

# --- 在「函式外面」做轉圈圈特效 ---
with st.spinner("📦 系統啟動中，正在載入保險資料庫..."):
    vectorstore = load_resources()

# --- 根據結果顯示 UI ---
if not vectorstore:
    st.error("❌ 資料庫載入失敗！請檢查 Requirements 或 Google Drive 連結。")
    st.stop()
else:
    # 成功載入後，偷偷給個小提示 (這是安全的，因為不在 cache 函式裡)
    st.toast("✅ 資料庫載入成功！", icon="🧠")

# 設定檢索器 (k=8 擴大搜尋範圍)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# ==========================================
# 5. 設定 LLM
# ==========================================
llm = ChatGroq(
    api_key=api_key,
    model="llama-3.1-8b-instant",
    temperature=0.3,
)

# ==========================================
# 6. Prompt 與 Chain
# ==========================================
persona_instruction = """
你是專業、靈活且富有洞察力的資深保險顧問。
你的任務是根據【已知資訊】(Context) 來回答使用者的問題或進行商品推薦。

🔥 **重要思考邏輯 (Chain of Thought)**：
1. **關鍵字轉換**：若使用者提到特定國家(如日本、美國)，請自動對應到條款中的「海外」、「國外」或「全球」相關規定。不要因為沒看到國家名字就說不知道。
2. **資訊整合**：若使用者詢問推薦，請綜合分析【已知資訊】中的多個商品，比較其優缺點。
3. **誠實但積極**：如果資料庫真的完全沒有相關險種，才回答無法提供；否則請盡量從現有資料中挖掘最接近的答案。

【已知資訊】：
{context}

使用者問題：{question}

請以台灣繁體中文，專業且條理分明地回答：
"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("human", persona_instruction)
])

def format_docs(docs):
    return "\n\n".join(f"文件來源: {doc.metadata.get('source', '未知')}\n內容: {doc.page_content}" for doc in docs)

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
# 7. 介面功能 (含 Debug 視窗)
# ==========================================
tab1, tab2 = st.tabs(["💬 線上保險諮詢", "📋 智能保險推薦"])

with tab1:
    st.subheader("💬 深度保險諮詢 (V11.0 資深理賠顧問版)")
    
    # 初始化歷史訊息
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # 顯示歷史對話
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 處理使用者輸入
    if user_input := st.chat_input("請輸入您的問題 (例如：手術險有包含門診手術嗎？)..."):
        
        # 1. 顯示使用者問題
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("🧠 顧問正在調閱過往對話與條款細節..."):
                try:
                    # ==========================================
                    # 🔧 引擎 1：對話歷史重組 (解決「金魚腦」問題)
                    # ==========================================
                    # 只有當有歷史對話時才執行重寫
                    search_query = user_input
                    history_context = ""
                    
                    if len(st.session_state.messages) > 1:
                        # 取出最近 4 句對話當作背景
                        recent_history = st.session_state.messages[-5:-1] 
                        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in recent_history])
                        
                        # 建立一個輕量級 LLM 來負責「翻譯」問題
                        llm_rewriter = ChatGroq(api_key=api_key, model="llama-3.1-8b-instant", temperature=0.1)
                        
                        rewrite_prompt = f"""
                        你是搜尋優化專家。請根據【對話歷史】將使用者的【最新問題】改寫成一個完整、獨立的搜尋語句。
                        
                        【對話歷史】：
                        {history_text}
                        
                        【最新問題】：
                        {user_input}
                        
                        【任務】：
                        如果最新問題依賴於歷史(例如只說「那併發症呢?」)，請補全主詞(例如「癌症險的併發症是否理賠?」)。
                        如果最新問題已經很完整，請直接輸出原句。
                        **只輸出改寫後的句子，不要有任何解釋。**
                        """
                        # 執行重寫
                        search_query = llm_rewriter.invoke(rewrite_prompt).content
                        # (選擇性) 可以在這裡 print 出來 debug
                        # print(f"原始問題: {user_input} -> 優化搜尋: {search_query}")

                    # ==========================================
                    # 🔧 引擎 2：核保級強力搜尋 (同步 Tab 2 規格)
                    # ==========================================
                    retriever_expert = vectorstore.as_retriever(
                        search_type="mmr", 
                        search_kwargs={"k": 6, "fetch_k": 1000, "lambda_mult": 0.5}
                    )
                    
                    # 使用「優化後的句子」去搜尋，精準度會大增
                    retrieved_docs = retriever_expert.invoke(search_query)

                    # Debug 視窗：讓您看到 AI 到底查了什麼
                    with st.expander(f"🕵️ [理賠視角] 搜尋語句：「{search_query}」"):
                        if not retrieved_docs:
                            st.warning("⚠️ 查無相關條款，請嘗試更具體的關鍵字。")
                        for i, doc in enumerate(retrieved_docs):
                            source = doc.metadata.get('source', '未知')
                            company = doc.metadata.get('company', '未知公司')
                            st.markdown(f"**{i+1}. [{company}] {source}**")
                            st.caption(doc.page_content[:150] + "...")

                    # ==========================================
                    # 🔧 引擎 3：理賠顧問 Prompt (Chain of Thought)
                    # ==========================================
                    # 使用低溫模型，確保回答嚴謹
                    llm_advisor = ChatGroq(api_key=api_key, model="llama-3.1-8b-instant", temperature=0.1)

                    persona_prompt = """
                    你是具備 20 年經驗的「資深保險理賠顧問」。
                    你的工作不是只有讀條款，而是要幫客戶「解釋條款背後的邏輯」與「實務理賠眉角」。

                    【已知條款資訊】：
                    {context}

                    【使用者問題】：
                    {question}

                    【回答策略 (請嚴格遵守)】：
                    1. **直球對決**：第一句話直接回答 Yes/No 或重點結論。
                    2. **條款依據**：引用條款中的關鍵字 (例如：「根據第 X 條...」)。
                    3. **名詞解釋**：如果條款有專有名詞 (如「既往症」、「等待期」)，請用白話文解釋給客戶聽。
                    4. **🚨 專家警示 (除外責任)**：這是最重要的！請主動告知**「什麼情況下不賠」**。專家就是要能看到陷阱。
                    5. **舉例說明**：請設計一個簡短的情境 (例如：小明發生了...) 來輔助說明。
                    6. **資料來源**：請在最後註明參考了哪一份文件。

                    請用台灣繁體中文，以專業、詳盡且有溫度的口吻回答：
                    """

                    qa_chain_expert = ChatPromptTemplate.from_template(persona_prompt) | llm_advisor | StrOutputParser()

                    # 準備 Context
                    docs_text = "\n\n".join(f"來源: {d.metadata.get('source', '未知')}\n內容: {d.page_content}" for d in retrieved_docs)
                    
                    # 生成回答
                    response = qa_chain_expert.invoke({"context": docs_text, "question": user_input})
                    
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

                except Exception as e:
                    st.error(f"發生錯誤：{e}")

with tab2:
    st.subheader("📋 全方位智能保險規劃書 (V10.0 核保風險評估版)")
    
    # --- 1. KYC (Know Your Customer) ---
    with st.container(border=True):
        st.markdown("#### 👤 第一步：建立您的風險與健康檔案")
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("性別", ["男", "女"])
            age = st.number_input("實歲年齡", 0, 100, 30)
            job = st.text_input("職業", "軟體工程師", help="請盡量詳細，影響意外險費率與職業等級")
            
            # 🔥 新增：生活習慣 (影響壽險/醫療險費率)
            lifestyle = st.multiselect("生活習慣 (影響費率)", ["吸菸", "飲酒", "嚼檳榔", "規律運動", "無不良嗜好"], default=["無不良嗜好"])
            
        with col2:
            salary = st.selectbox("年收入", ["50萬以下", "50-100萬", "100-200萬", "200萬以上"])
            family_status = st.selectbox("家庭責任", ["單身未婚", "已婚無子", "已婚有子 (小孩幼齡)", "已婚有子 (小孩已獨立)", "單親家庭"])
            
            # 🔥 新增：健康告知 (這是經理人最在意的核保關鍵！)
            health_history = st.multiselect("過往病史/健康狀況 (核保關鍵)", 
                ["無", "高血壓", "糖尿病", "心臟疾病", "兩年內曾住院/手術", "五年內曾罹患癌症", "領有重大傷病卡"],
                default=["無"],
                help="請誠實告知，這將影響 AI 對於「除外責任」或「加費」的判斷"
            )

        st.markdown("#### 🛡️ 第二步：您的保險需求與預算")
        col3, col4 = st.columns(2)
        with col3:
            ins_type = st.selectbox("想規劃的險種", 
                ["壽險 (定期/終身)", "醫療險 (實支實付/日額)", "意外險 (傷害保險)", "重大傷病/癌症險", "儲蓄/理財險", "旅遊平安險"]
            )
        with col4:
            st.markdown("💰 **預算設定**")
            b_col1, b_col2 = st.columns([1, 1])
            with b_col1:
                budget_amount = st.number_input("金額", min_value=0, value=None, step=500, placeholder="請輸入金額")
            with b_col2:
                budget_period = st.selectbox("繳費頻率", ["月繳", "年繳", "躉繳(一次付清)"])

        # 特殊欄位
        extra_info = ""
        if "旅遊" in ins_type:
            dest = st.text_input("旅遊國家", "日本")
            days = st.number_input("天數", 1, 365, 5)
            extra_info = f"預計前往{dest}旅遊{days}天"
        
        has_insurance = st.checkbox("我已有類似保險")
        extra_info += "。已有類似保單，重點在補強。" if has_insurance else "。新投保。"

    # --- 2. 開始分析 ---
    if st.button("🚀 啟動核保級分析", type="primary"):
        
        # 防呆
        if budget_amount is None or budget_amount == 0:
            st.warning("⚠️ 請輸入預算金額。")
            st.stop()
        
        total_annual_budget = budget_amount * 12 if budget_period == "月繳" else budget_amount
        budget_desc = f"{budget_period} {budget_amount} 元 (年繳約 {total_annual_budget} 元)"

        if "旅遊" not in ins_type and total_annual_budget < 2000:
            st.error("❌ 預算過低，無法規劃有效的主約商品。")
            st.stop()

        with st.spinner("🤖 AI 正在進行核保風險評估與條款比對 (fetch_k=1000)..."):
            
            # 維持 V9 的強力搜尋
            retriever_manager = vectorstore.as_retriever(
                search_type="mmr", 
                search_kwargs={"k": 6, "fetch_k": 1000, "lambda_mult": 0.5}
            )

            search_keyword = f"{ins_type} 條款 保單"
            if "旅遊" in ins_type:
                search_keyword += f" {dest}"

            retrieved_docs = retriever_manager.invoke(search_keyword)

            with st.expander("🕵️ [工程師模式] 檢索到的候選名單"):
                if not retrieved_docs:
                    st.warning("⚠️ 無法檢索到相關條款。")
                for i, doc in enumerate(retrieved_docs):
                    source = doc.metadata.get('source', doc.metadata.get('filename', '未知'))
                    company = doc.metadata.get('company', '未知公司')
                    st.markdown(f"**{i+1}. [{company}] {source}**")
                    st.caption(doc.page_content[:100] + "...")

            # 低溫模型
            llm_strict = ChatGroq(
                api_key=api_key,
                model="llama-3.1-8b-instant",
                temperature=0.2 
            )

            # ==========================================
            # 🔥 V10 核心：核保邏輯 Prompt
            # ==========================================
            query = f"""
            【客戶畫像 (KYC)】：
            - 基本資料：{gender}, {age}歲, 職業：{job}
            - 預算：{budget_desc} (嚴格遵守)
            - 家庭責任：{family_status}
            - **健康狀況 (核保關鍵)**：{', '.join(health_history)} (🔥若有病史，請注意除外責任或拒保風險)
            - **生活習慣**：{', '.join(lifestyle)} (🔥若有吸菸，壽險費率可能增加)
            - 需求目標：{ins_type}
            - 備註：{extra_info}

            【任務指令】：
            你是資深的「核保人員」兼「保險經紀人」。請閱讀檢索資料，產出專業建議書。

            1. **核保風險預判**：
               - 若客戶有「糖尿病/高血壓」等病史，請在推薦時明確警告：「此體況可能面臨加費、除外或拒保」。
               - 若客戶是「高風險職業」，請檢查意外險條款是否承保。
            
            2. **深度推薦理由 (Deep Reasoning)**：
               - 禁止只寫「這張很好」。
               - 必須寫出邏輯：**「因為您是 [A身份/有B體況]，這張保單的 [C條款] 對您有利，且符合您的 [D預算]。」**
            
            3. **精選雙商品比較**：挑選 2 個方案 (嘗試不同公司)。

            【建議書輸出格式】：
            ### 🩺 第一部分：核保風險評估
            (針對客戶的健康、職業與生活習慣，預判投保可能遇到的阻礙或加費狀況)

            ### 🏆 第二部分：精選方案推薦
            #### 方案 A：[保險公司] - [商品名稱]
            * **核心優勢**：(一句話亮點)
            * **深度推薦原因**：(🔥請依照「使用者特徵 + 條款細節 + 解決痛點」的邏輯撰寫)
            * **核保注意事項**：(針對該商品的職業或體況限制)
            * **資料來源**：(請註明參考文件)

            #### 方案 B：[保險公司] - [商品名稱]
            * **核心優勢**：...
            * **深度推薦原因**：...
            * **核保注意事項**：...
            * **資料來源**：...

            ### ⚖️ 第三部分：超級比一比
            | 比較項目 | 方案 A | 方案 B |
            | :--- | :--- | :--- |
            | 保險公司 | ... | ... |
            | 商品特色 | ... | ... |
            | 承保範圍 | ... | ... |
            | **預估保費** | (依 {budget_period} 估算) | (依 {budget_period} 估算) |

            ### 💡 經理人總結
            (給客戶的最終建議)
            """
            
            try:
                docs_text = "\n\n".join(f"來源: {d.metadata.get('source', '未知')}\n內容: {d.page_content}" for d in retrieved_docs)
                
                prompt_template = ChatPromptTemplate.from_template(query + "\n\n【檢索到的條款內容】：\n{context}")
                chain = prompt_template | llm_strict | StrOutputParser()
                
                response = chain.invoke({"context": docs_text})
                st.markdown(response)

                st.info("💡 **經理人小叮嚀**：\n本建議書由 AI 系統生成。若您有「過往病史」，實際核保結果（加費/除外/拒保）將由保險公司核保科最終決定。")
                
            except Exception as e:
                st.error(f"分析過程發生錯誤: {e}")