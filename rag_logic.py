# rag_logic.py

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# --- 상수 정의 ---
# 현재 스크립트와 같은 디렉토리에 PDF 파일이 있도록 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(BASE_DIR, "주간농사정보 제28호.pdf") #PDF_PATH = os.path.join(BASE_DIR, "주간농사정보 제28호.pdf")

EMBEDDING_MODEL_NAME = "nomic-embed-text"
LLM_MODEL_NAME = "gemma3"  # llama3로 변경 가능


# RAG 체인을 초기화하고 반환하는 함수
def initialize_rag_chain():
    """
    문서 로드, 분할, 임베딩, 벡터DB 생성, RAG 체인 생성을 수행하고
    생성된 RAG 체인을 반환합니다.
    """
    print("--- RAG 파이프라인 초기화 시작 ---")
    
    # 파일 존재 여부 확인
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {PDF_PATH}")
    
    print(f"📄 PDF 파일 경로: {PDF_PATH}")
    
    # 1. 문서 로드
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()
    print("✅ [1/5] 문서 로드 완료")
    
    # 2. 문서 분할
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    print("✅ [2/5] 문서 분할 완료")
    
    # 3. 임베딩 모델 및 벡터 DB 설정
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    print("✅ [3/5] FAISS 벡터 DB 생성 완료")
    
    # 4. 검색기 생성
    retriever = vectorstore.as_retriever()
    print("✅ [4/5] 검색기 생성 완료")
    
    # 5. 프롬프트 및 LLM 설정, RAG 체인 생성
    template = """당신은 주어진 문맥(context)의 내용을 바탕으로만 질문에 답하는 AI 어시스턴트입니다.
사실에 기반하여 정확하고 간결하게 답변해주세요. 문맥에서 답을 찾을 수 없다면 "문맥에서 정보를 찾을 수 없습니다."라고 답하세요. 모든 답변은 한국어로 대답해줘.

CONTEXT: {context}

QUESTION: {question}
"""
    
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOllama(model=LLM_MODEL_NAME, temperature=0)
    
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print("✅ [5/5] RAG 체인 생성 완료")
    print("--- RAG 파이프라인 초기화 완료 ---")
    
    return rag_chain

# 질문에 대한 답변을 가져오는 함수
def get_answer(chain, question):
    """
    초기화된 RAG 체인과 질문을 받아 답변을 반환합니다.
    """
    return chain.invoke(question)
