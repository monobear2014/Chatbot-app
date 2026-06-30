import streamlit as st
import ollama

model = "vicuna:7b-v1.5-q5_1"
st.title("Simple Chatbot App")
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    st.chat_message(m["role"]).write(m["content"])

if prompt := st.chat_input("Nhập câu hỏi ..."):
    st.session_state.messages.append({
        "role": "user", "content": prompt,
    })
    st.chat_message("user").write(prompt)
    resp = ollama.chat(
        model=model,
        messages=st.session_state.messages,
        options={"temperature": 0},
    )
    answer = resp.message.content
    st.session_state.messages.append({
        "role": "assistant", "content": answer,
    })
    st.chat_message("assistant").write(answer)