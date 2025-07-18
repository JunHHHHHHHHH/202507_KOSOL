# app.py
import streamlit as st
from rag_logic import initialize_rag_chain, get_answer

st.set_page_config(page_title="RAG 챗봇", page_icon="🤖")
st.title("🤖 문서 기반 RAG 챗봇")

# OpenAI API 키 입력
st.sidebar.title("🔑 API 설정")
openai_api_key = st.sidebar.text_input(
    "OpenAI API 키를 입력하세요:",
    type="password",
    placeholder="sk-..."
)

if not openai_api_key:
    st.warning("⚠️ OpenAI API 키를 입력해주세요.")
    st.stop()

# RAG 체인 초기화
if "rag_chain" not in st.session_state or st.session_state.get("api_key") != openai_api_key:
    try:
        with st.spinner("초기화 중..."):
            st.session_state.rag_chain = initialize_rag_chain(openai_api_key)
            st.session_state.api_key = openai_api_key
        st.success("✅ 초기화 완료!")
    except Exception as e:
        st.error(f"❌ 초기화 오류: {str(e)}")
        st.stop()

# 채팅 기능
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("질문을 입력하세요"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        try:
            with st.spinner("답변 생성 중..."):
                response = get_answer(st.session_state.rag_chain, prompt)
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"❌ 답변 생성 오류: {str(e)}")
