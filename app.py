# app.py

import streamlit as st
from rag_logic import initialize_rag_chain, get_answer

# --- Streamlit 앱 설정 ---
st.set_page_config(page_title="RAG 챗봇", page_icon="🤖")
st.title("🤖 문서 기반 RAG 챗봇")

# --- OpenAI API 키 입력 ---
st.sidebar.title("🔑 API 설정")
openai_api_key = st.sidebar.text_input(
    "OpenAI API 키를 입력하세요:",
    type="password",
    placeholder="sk-..."
)

# API 키가 없으면 경고 메시지 표시
if not openai_api_key:
    st.warning("⚠️ OpenAI API 키를 입력해주세요. 사이드바에서 API 키를 입력하면 챗봇을 사용할 수 있습니다.")
    st.stop()

# --- RAG 체인 초기화 및 세션 상태에 저장 ---
if "rag_chain" not in st.session_state or st.session_state.get("api_key") != openai_api_key:
    st.write("RAG 파이프라인을 초기화하고 있습니다. 잠시만 기다려주세요...")
    
    try:
        with st.spinner("문서 로딩, 임베딩, 벡터 DB 생성 중..."):
            st.session_state.rag_chain = initialize_rag_chain(openai_api_key)
            st.session_state.api_key = openai_api_key
        st.success("✅ 초기화 완료! 이제 질문을 입력하세요.")
    except Exception as e:
        st.error(f"❌ 초기화 중 오류가 발생했습니다: {str(e)}")
        st.stop()

# --- 채팅 기록 관리 ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# 이전 대화 내용 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 사용자 입력 및 챗봇 응답 처리 ---
if prompt := st.chat_input("질문을 입력해 주세요."):
    # 1. 사용자 메시지를 채팅 기록에 추가하고 화면에 표시
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 2. 챗봇의 답변 생성
    with st.chat_message("assistant"):
        try:
            with st.spinner("답변을 생성하는 중입니다..."):
                response = get_answer(st.session_state.rag_chain, prompt)
                st.markdown(response)
            
            # 3. 챗봇 메시지를 채팅 기록에 추가
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"❌ 답변 생성 중 오류가 발생했습니다: {str(e)}")
