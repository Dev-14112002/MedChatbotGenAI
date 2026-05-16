import streamlit as st

st.title("🩺 Medical Chatbot")

user_input = st.chat_input("Ask a medical question")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        st.markdown("Hello! I am your medical assistant.")
