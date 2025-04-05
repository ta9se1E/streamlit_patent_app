import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.graph import StateGraph
from typing import List, Annotated, Literal, Sequence, TypedDict
from langgraph.graph import END, StateGraph, START
import asyncio
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import Document
from dotenv import load_dotenv
from langsmith import Client
import streamlit_authenticator as stauth
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import japanize_matplotlib 
import numpy as np
import seaborn as sns
from st_login_form import login_form
from datetime import datetime
from janome.tokenizer import Tokenizer
from wordcloud import WordCloud

# フォントパスの指定（必要に応じて利用）
font_path1 = "./font/NotoSansJP-Regular.ttf"

# .envファイルを読み込む
load_dotenv(dotenv_path=".env.example")

# パスワードを環境変数から取得
PASSWORD = os.getenv("PASSWORD")

# パスワード入力ボックス
inputText_A = st.text_input('パスワードを入力してください', type="password")

# 環境変数`PASSWORD`と比較
if inputText_A == PASSWORD:
        
    # ---- ログイン成功後に表示するアプリのメイン機能をここに記述 ----
    st.write("ここにチャットシステムやデータ分析モードなどを切り替えるコードを実装できます。")

    # --- モード選択 ---
    mode = st.sidebar.radio("モード選択", ("チャットシステム", "データ解析システム"))

    if mode == "チャットシステム":
        
        load_dotenv(dotenv_path ="langchain-book/test/.env.example")

        class RouteQuery(BaseModel):
            """ユーザーのクエリを最も関連性の高いデータソースにルーティングします。"""

            datasource: Literal["vectorstore", "web_search"] = Field(
                ...,
                description="ユーザーの質問に応じて、ウェブ検索またはベクターストアにルーティングします。",
            )


        class GradeDocuments(BaseModel):
            """取得された文書の関連性チェックのためのバイナリスコア。"""

            binary_score: str = Field(
                description="文書が質問に関連しているかどうか、「yes」または「no」"
            )


        class GradeHallucinations(BaseModel):
            """生成された回答における幻覚の有無を示すバイナリスコア。"""

            binary_score: str = Field(
                description="回答が事実に基づいているかどうか、「yes」または「no」"
            )

        class GradeAnswer(BaseModel):
            """回答が質問に対処しているかどうかを評価するバイナリスコア。"""

            binary_score: str = Field(
                description="回答が質問に対処しているかどうか、「yes」または「no」"
            )

        class GraphState(TypedDict):
            """
            グラフの状態を表します。

            属性:
                question: 質問
                generation: LLM生成
                documents: 文書のリスト
            """

            question: str
            generation: str
            documents: List[str]

        async def route_question(state):
            st.session_state.status.update(label=f"**---ROUTE QUESTION---**", state="running", expanded=True)
            st.session_state.log += "---ROUTE QUESTION---" + "\n\n"
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            structured_llm_router = llm.with_structured_output(RouteQuery)

            system = """あなたはユーザーの質問をベクターストアまたはウェブ検索にルーティングする専門家です。
        ベクターストアには圧力容器の製造方法や関連技術に関連する特許文書が含まれています。
        これらのトピックに関する質問にはベクターストアを使用し、それ以外の場合はウェブ検索を使用してください。"""
            route_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system),
                    ("human", "{question}"),
                ]
            )

            question_router = route_prompt | structured_llm_router

            question = state["question"]
            source = question_router.invoke({"question": question})
            if source.datasource == "web_search":
                st.session_state.log += "---ROUTE QUESTION TO WEB SEARCH---" + "\n\n"
                st.session_state.placeholder.markdown("---ROUTE QUESTION TO WEB SEARCH---")
                return "web_search"
            elif source.datasource == "vectorstore":
                st.session_state.placeholder.markdown("ROUTE QUESTION TO RAG")
                st.session_state.log += "ROUTE QUESTION TO RAG" + "\n\n"
                return "vectorstore"

        async def retrieve(state):
            st.session_state.status.update(label=f"**---RETRIEVE---**", state="running", expanded=True)
            st.session_state.placeholder.markdown(f"RETRIEVING…\n\nKEY WORD:{state['question']}")
            st.session_state.log += f"RETRIEVING…\n\nKEY WORD:{state['question']}" + "\n\n"

            
            #編集ポイント
            embd = OpenAIEmbeddings(model="text-embedding-3-small")


            from langchain_community.document_loaders import PyPDFDirectoryLoader
            from langchain.text_splitter import CharacterTextSplitter 
            from langchain.vectorstores import FAISS 
            from langchain.document_loaders import PyPDFLoader 
            
            retriever = FAISS.load_local("vectorstore_20250404", 
                                                    embd,
                                                    allow_dangerous_deserialization=True)
            #編集終わり

            question = state["question"]
            documents = retriever.similarity_search(question)
            
            for doc in documents:
                if "source" in doc.metadata:
                    doc.page_content += f"\n\nSource: {doc.metadata['source']}"
                else:
                    doc.page_content += "\n\nSource: Unknown"

            
            st.session_state.placeholder.markdown("RETRIEVE SUCCESS!!")
            return {"documents": documents, "question": question}

        async def web_search(state):
            st.session_state.status.update(label=f"**---WEB SEARCH---**", state="running", expanded=True)
            st.session_state.placeholder.markdown(f"WEB SEARCH…\n\nKEY WORD:{state['question']}")
            st.session_state.log += f"WEB SEARCH…\n\nKEY WORD:{state['question']}" + "\n\n"
            
            TAVILY_API_KEY = os.getenv("tavily_api_key")

            question = state["question"]
            web_search_tool = TavilySearchResults(k=3)

            docs = web_search_tool.invoke({"query": question})
            web_results = []
            
            #追加
            for doc in docs:
                content_with_source = doc["content"] + f"\n\nSource: {doc.get('source', 'Unknown')}"
                web_results.append(Document(page_content=content_with_source, metadata={"source": doc.get('source', 'Unknown')}))
            
            return {"documents": web_results, "question": question}


        async def grade_documents(state):
            st.session_state.number_trial += 1
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            structured_llm_grader = llm.with_structured_output(GradeDocuments)

            system = """あなたは、ユーザーの質問に対して取得されたドキュメントの関連性を評価する採点者です。
        ドキュメントにユーザーの質問に関連するキーワードや意味が含まれている場合、それを関連性があると評価してください。
        目的は明らかに誤った取得を排除することです。厳密なテストである必要はありません。
        ドキュメントが質問に関連しているかどうかを示すために、バイナリスコア「yes」または「no」を与えてください。"""
            grade_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system),
                    ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
                ]
            )

            retrieval_grader = grade_prompt | structured_llm_grader
            st.session_state.status.update(label=f"**---CHECK DOCUMENT RELEVANCE TO QUESTION---**", state="running", expanded=False)
            st.session_state.log += "**---CHECK DOCUMENT RELEVANCE TO QUESTION---**" + "\n\n"
            question = state["question"]
            documents = state["documents"]
            filtered_docs = []
            i = 0
            for d in documents:
                if st.session_state.number_trial <= 2:
                    file_name = d.metadata["source"]
                    file_name = os.path.basename(file_name.replace("\\","/"))
                    i += 1
                    score = retrieval_grader.invoke(
                        {"question": question, "document": d.page_content}
                    )
                    grade = score.binary_score
                    if grade == "yes":
                        st.session_state.status.update(label=f"**---GRADE: DOCUMENT RELEVANT---**", state="running", expanded=True)
                        st.session_state.placeholder.markdown(f"DOC {i}/{len(documents)} : **RELEVANT**\n\n")
                        st.session_state.log += "---GRADE: DOCUMENT RELEVANT---" + "\n\n"
                        st.session_state.log += f"doc {i}/{len(documents)} : RELEVANT\n\n"
                        filtered_docs.append(d)
                    else:
                        st.session_state.status.update(label=f"**---GRADE: DOCUMENT NOT RELEVANT---**", state="error", expanded=True)
                        st.session_state.placeholder.markdown(f"DOC {i}/{len(documents)} : **NOT RELEVANT**\n\n")
                        st.session_state.log += "---GRADE: DOCUMENT NOT RELEVANT---" + "\n\n"
                        st.session_state.log += f"DOC {i}/{len(documents)} : NOT RELEVANT\n\n"
                else:

                    filtered_docs.append(d)

            if not st.session_state.number_trial <= 2:
                st.session_state.status.update(label=f"**---NO NEED TO CHECK---**", state="running", expanded=True)
                st.session_state.placeholder.markdown("QUERY TRANSFORMATION HAS BEEN COMPLETED")
                st.session_state.log += "QUERY TRANSFORMATION HAS BEEN COMPLETED" + "\n\n"

            return {"documents": filtered_docs, "question": question}

        async def generate(state):
            st.session_state.status.update(label=f"**---GENERATE---**", state="running", expanded=False)
            st.session_state.log += "---GENERATE---" + "\n\n"
            prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", """ユーザーから与えられたコンテキストを参考に質問に対し答えて下さい。"""),
                        ("human", """Question: {question} 
        Context: {context}"""),
                    ]
                )
                
            llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

            rag_chain = prompt | llm | StrOutputParser()
            question = state["question"]
            documents = state["documents"]
            generation = rag_chain.invoke({"context": documents, "question": question})
            return {"documents": documents, "question": question, "generation": generation}


        async def transform_query(state):
            st.session_state.status.update(label=f"**---TRANSFORM QUERY---**", state="running", expanded=True)
            st.session_state.placeholder.empty()
            st.session_state.log += "---TRANSFORM QUERY---" + "\n\n"
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

            system = """あなたは、入力された質問をベクトルストア検索に最適化されたより良いバージョンに変換する質問リライターです。
        質問を見て、質問者の意図/意味について推論してより良いベクトル検索の為の質問を作成してください。"""
            re_write_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system),
                    (
                        "human",
                        "Here is the initial question: \n\n {question} \n Formulate an improved question.",
                    ),
                ]
            )

            question_rewriter = re_write_prompt | llm | StrOutputParser()
            question = state["question"]
            documents = state["documents"]
            better_question = question_rewriter.invoke({"question": question})
            st.session_state.log += f"better_question : {better_question}\n\n"
            st.session_state.placeholder.markdown(f"better_question : {better_question}")
            return {"documents": documents, "question": better_question}


        async def decide_to_generate(state):
            filtered_documents = state["documents"]
            if not filtered_documents:
                st.session_state.status.update(label=f"**---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---**", state="error", expanded=False)
                st.session_state.log += "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---" + "\n\n"
                return "transform_query"                                     
            else:
                st.session_state.status.update(label=f"**---DECISION: GENERATE---**", state="running", expanded=False)
                st.session_state.log += "---DECISION: GENERATE---" + "\n\n"
                return "generate"

        async def grade_generation_v_documents_and_question(state):
            st.session_state.number_trial += 1
            st.session_state.status.update(label=f"**---CHECK HALLUCINATIONS---**", state="running", expanded=False)
            st.session_state.log += "---CHECK HALLUCINATIONS---" + "\n\n"
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            structured_llm_grader = llm.with_structured_output(GradeHallucinations)

            system = """あなたは、LLMの生成が取得された事実のセットに基づいているか/サポートされているかを評価する採点者です。
        バイナリスコア「yes」または「no」を与えてください。「yes」は、回答が事実のセットに基づいている/サポートされていることを意味します。"""
            hallucination_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system),
                    ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
                ]
            )
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            structured_llm_grader = llm.with_structured_output(GradeAnswer)

            system = """あなたは、回答が質問に対処しているか/解決しているかを評価する採点者です。
        バイナリスコア「yes」または「no」を与えてください。「yes」は、回答が質問を解決していることを意味します。"""
            answer_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system),
                    ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
                ]
            )

            answer_grader = answer_prompt | structured_llm_grader
            hallucination_grader = hallucination_prompt | structured_llm_grader
            question = state["question"]
            documents = state["documents"]
            generation = state["generation"]
            score = hallucination_grader.invoke(
                {"documents": documents, "generation": generation}
            )
            grade = score.binary_score
            if st.session_state.number_trial <= 3:
                if grade == "yes":
                    st.session_state.placeholder.markdown("DECISION: ANSWER IS BASED ON A SET OF FACTS")
                    st.session_state.log += "---DECISION: ANSWER IS BASED ON A SET OF FACTS---" + "\n\n"
                    st.session_state.log += "---GRADE GENERATION vs QUESTION---" + "\n\n"
                    score = answer_grader.invoke({"question": question, "generation": generation})
                    grade = score.binary_score
                    st.session_state.status.update(label=f"**---GRADE GENERATION vs QUESTION---**", state="running", expanded=True)
                    if grade == "yes":
                        st.session_state.status.update(label=f"**---DECISION: GENERATION ADDRESSES QUESTION---**", state="running", expanded=True)
                        with st.session_state.placeholder:
                            st.markdown("**USEFUL!!**")
                            st.markdown(f"question : {question}")
                            st.markdown(f"generation : {generation}")                   
                            st.session_state.log += "---DECISION: GENERATION ADDRESSES QUESTION---" + "\n\n"
                            st.session_state.log += f"USEFUL!!\n\n"
                            st.session_state.log += f"question:{question}\n\n"
                            st.session_state.log += f"generation:{generation}\n\n"
                        return "useful"
                    else:
                        st.session_state.number_trial -= 1
                        st.session_state.status.update(label=f"**---DECISION: GENERATION DOES NOT ADDRESS QUESTION---**", state="error", expanded=True)
                        with st.session_state.placeholder:
                            st.markdown("**NOT USEFUL**")
                            st.markdown(f"question:{question}")
                            st.markdown(f"generation:{generation}")
                            st.session_state.log += "---DECISION: GENERATION DOES NOT ADDRESS QUESTION---" + "\n\n"
                            st.session_state.log += f"NOT USEFUL\n\n"
                            st.session_state.log += f"question:{question}\n\n"
                            st.session_state.log += f"generation:{generation}\n\n"
                        return "not useful"
                else:
                    st.session_state.status.update(label=f"**---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---**", state="error", expanded=True)
                    with st.session_state.placeholder:
                        st.markdown("not grounded")
                        st.markdown(f"question:{question}")
                        st.markdown(f"generation:{generation}")
                        st.session_state.log += "---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---" + "\n\n"
                        st.session_state.log += f"not grounded\n\n"
                        st.session_state.log += f"question:{question}\n\n"
                        st.session_state.log += f"generation:{generation}\n\n"
                    return "not supported"
            else:
                st.session_state.status.update(label=f"**---NO NEED TO CHECK---**", state="running", expanded=True)
                st.session_state.placeholder.markdown("TRIAL LIMIT EXCEEDED")
                st.session_state.log += "---NO NEED TO CHECK---" + "\n\n"
                st.session_state.log += "TRIAL LIMIT EXCEEDED" + "\n\n"
                return "useful"
            
        # 初期化→追加
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []    

        # チャット履歴を更新→追加
        async def run_workflow(inputs):
            st.session_state.number_trial = 0
            with st.status(label="**GO!!**", expanded=True, state="running") as st.session_state.status:
                st.session_state.placeholder = st.empty()
                value = await st.session_state.workflow.ainvoke(inputs)
            
            # チャット履歴に追加
            st.session_state.chat_history.append({
                "question": inputs["question"],
                "response": value["generation"],
                "documents": value["documents"]  # ソース情報を含むドキュメント
            })

            st.session_state.placeholder.empty()
            st.session_state.message_placeholder = st.empty()
            st.session_state.status.update(label="**FINISH!!**", state="complete", expanded=False)
            st.session_state.message_placeholder.markdown(value["generation"])
            with st.popover("ログ"):
                st.markdown(st.session_state.log)

        if st.button("Show Chat History"):
            for i, entry in enumerate(st.session_state.chat_history):
                st.markdown(f"### Message {i+1}")
                st.markdown(f"**Question:** {entry['question']}")
                st.markdown(f"**Response:** {entry['response']}")
                st.markdown("**Sources:**")
                for doc in entry["documents"]:
                    source_info = doc.metadata.get("source", "Unknown")
                    st.markdown(f"- {source_info}")

        def st_rag_langgraph():
            
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
            LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
            os.environ["LANGCHAIN_PROJECT"] =  "Carbon-GPTs"

            client = Client(api_key=LANGSMITH_API_KEY)

            if 'log' not in st.session_state:
                st.session_state.log = ""

            if 'status_container' not in st.session_state:
                st.session_state.status_container = st.empty()

            if not hasattr(st.session_state, "workflow"):

                workflow = StateGraph(GraphState)

                workflow.add_node("web_search", web_search)
                workflow.add_node("retrieve", retrieve)
                workflow.add_node("grade_documents", grade_documents)
                workflow.add_node("generate", generate)
                workflow.add_node("transform_query", transform_query)

                workflow.add_conditional_edges(
                    START,
                    route_question,
                    {
                        "vectorstore": "retrieve",
                        "web_search": "web_search",
                    },
                )
                workflow.add_edge("web_search", "generate")
                workflow.add_edge("retrieve", "grade_documents")
                workflow.add_conditional_edges(
                    "grade_documents",
                    decide_to_generate,
                    {
                        "transform_query": "transform_query",
                        "generate": "generate",
                    },
                )
                workflow.add_edge("transform_query", "retrieve")
                workflow.add_conditional_edges(
                    "generate",
                    grade_generation_v_documents_and_question,
                    {
                        "not supported": "generate",
                        "useful": END,
                        "not useful": "transform_query",
                    },
                )

                app = workflow.compile()
                app = app.with_config(recursion_limit=10,run_name="Agent",tags=["Agent"])
                app.name = "Agent"
                st.session_state.workflow = app


            st.title("Adaptive RAG by LangGraph")

            if prompt := st.chat_input("質問を入力してください"):
                st.session_state.log = ""
                with st.chat_message("user", avatar="😊"):
                    st.markdown(prompt)

                inputs = {"question": prompt}
                asyncio.run(run_workflow(inputs))
    
        if __name__ == "__main__":
            st_rag_langgraph()
        

    elif mode == "データ解析システム":
        st.title("データ解析システム")
        
        def st_rag_langgraph():
            st.write("CSVファイルをアップロードしてください。")
            uploaded_file = st.file_uploader("CSVファイルを選択", type=["csv"])
        
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file, index_col=0, header=0)
                df["出願日"] = pd.to_datetime(df["出願日"], errors='coerce')
                df["公開・公表日"] = pd.to_datetime(df["公開・公表日"], errors='coerce')
                df["出願年"] = df["出願日"].dt.year
                df["公開・公表年"] = df["公開・公表日"].dt.year
                df["出願件数"] = df["出願番号"].value_counts().values
                
                start_date = datetime(2000, 1, 1)
                end_date = datetime(2025, 3, 31)
                
                term = st.slider("期間",value=[start_date,end_date])
                
                _df = df[(df["出願日"] >= term[0]) & (df["出願日"] <= term[1])]
                
                st.write("データプレビュー:", _df.head())
                
               
                analyze_mode = st.sidebar.radio("分析モード選択", ("特許出願件数把握", "ネットワーク分析","ワードマップ"))
                
                if analyze_mode == "特許出願件数把握":
                    _ = pd.DataFrame(_df.groupby(["出願人・権利者(最新)","出願年"]).agg({"出願件数":"sum", "Unnamed: 8":"first"}))
                    fig, ax = plt.subplots(figsize=(25,12))
                    sns.barplot(data=_, x="出願年", y="出願件数", hue="Unnamed: 8", errorbar=None)
                    ax.set_xlabel("出願年", fontsize=40)
                    ax.set_ylabel("出願件数", fontsize=40)
                    ax.tick_params(axis='y', labelsize=15)
                    ax.tick_params(axis='x', labelsize=15)
                    st.pyplot(fig)
                    
                elif analyze_mode == "ネットワーク分析":
                                        
                    # 欠損値を削除
                    df_clean = _df.dropna(subset=["出願人・権利者(最新)"])

                    # 出願人データをリストに変換（共同出願の場合、";"で分割）
                    edges = []
                    for applicants in df_clean["出願人・権利者(最新)"]:
                        applicant_list = [a.strip() for a in applicants.split(";")]
                        if len(applicant_list) > 1:  # 共同出願がある場合のみ
                            for i in range(len(applicant_list)):
                                for j in range(i + 1, len(applicant_list)):
                                    edges.append((applicant_list[i], applicant_list[j]))

                    # グラフ作成
                    G = nx.Graph()
                    G.add_edges_from(edges)

                    # 描画
                    plt.figure(figsize=(10, 7))
                    pos = nx.spring_layout(G, seed=42)  # レイアウト設定
                    nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=2000, font_size=10,
                        font_family="IPAexGothic"  # 日本語対応フォントを明示的に指定
                        )
                    plt.title("出願人・共同出願ネットワーク", fontsize=14)
                    # Streamlitでグラフを表示
                    st.pyplot(plt)
                    
                    df_clean = _df.dropna(subset=["発明者または考案者(最新)"])

                    # 出願人データをリストに変換（共同出願の場合、";"で分割）
                    edges = []
                    for applicants in df_clean["発明者または考案者(最新)"]:
                        applicant_list = [a.strip() for a in applicants.split(";")]
                        if len(applicant_list) > 1:  # 共同出願がある場合のみ
                            for i in range(len(applicant_list)):
                                for j in range(i + 1, len(applicant_list)):
                                    edges.append((applicant_list[i], applicant_list[j]))

                    # グラフ作成
                    G = nx.Graph()
                    G.add_edges_from(edges)

                    # 描画
                    plt.figure(figsize=(10, 7))
                    pos = nx.spring_layout(G, seed=42)  # レイアウト設定
                    nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=2000, font_size=10,
                        font_family="IPAexGothic"  # 日本語対応フォントを明示的に指定
                        )
                    plt.title("発明人ネットワーク", fontsize=14)
                    # Streamlitでグラフを表示
                    st.pyplot(plt)
                    
                if analyze_mode == "ワードマップ":
                    
                    # データから発明等の名称を取得
                    appnames = _df['発明等の名称'].values

                    # Janomeトークナイザを初期化
                    tokenizer = Tokenizer()
                    words = []

                    # 名詞の連続を検出して複合名詞として結合
                    for appname in appnames:
                        tokens = tokenizer.tokenize(appname)
                        noun_phrase = []  # 複合名詞用のリスト
                        for token in tokens:
                            if token.part_of_speech.startswith('名詞'):
                                noun_phrase.append(token.surface)  # 名詞を追加
                            else:
                                if noun_phrase:  # 名詞が連続していた場合、結合してリストに追加
                                    words.append("".join(noun_phrase))
                                    noun_phrase = []  # リセット
                        # 最後に残った名詞を追加
                        if noun_phrase:
                            words.append("".join(noun_phrase))

                    # 単語の出現頻度を計算
                    df_words = pd.Series(words).value_counts()
                    word_counts = df_words.to_dict()

                    # 単語の頻度を棒グラフで表示
                    fig,ax = plt.subplots(figsize=(20, 13))
                    head_20 = df_words.iloc[:20].copy()
                    ax.barh(y=head_20.index, width=head_20.values, color="orange")
                                        
                    # グラフのタイトルとラベル
                    ax.set_title("頻出単語のトップ20", fontsize=40)
                    ax.set_xlabel("頻度", fontsize=40)
                    ax.set_ylabel("単語", fontsize=40)

                    # y軸のラベルを調整
                    #ax.set_yticks(head_20.index)
                    #ax.set_yticklabels(head_20.index, rotation=0, fontsize=30)
                    ax.tick_params(axis='y', labelsize=30)
                    ax.tick_params(axis='x', labelsize=30)
                    
                    st.pyplot(fig)
                    
                    # 日本語フォントを指定
                    font_path1 = "./font/NotoSansJP-Regular.ttf"  # 適切なパスを指定

                    # ワードクラウドを生成
                    wordcloud = WordCloud(
                        background_color='white', 
                        width=800, 
                        height=600, 
                        font_path=font_path1
                    )

                    wordcloud.generate_from_frequencies(word_counts)

                    # ワードクラウドを表示
                    plt.figure(figsize=(10, 8))
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis('off')

                    # Streamlitで表示
                    st.pyplot(plt)
            
                
        if __name__ == "__main__":
            st_rag_langgraph()


