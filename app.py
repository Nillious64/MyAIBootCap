import streamlit as st
st.title("My AI app")
with st.sidebar:
    st.header("Settings")
    name = st.text_input("Enter your name")
    mood = st.selectbox("What will your AI's mood be?", ["Happy", "Sad", "Angry"])
    creativity = st.slider("Creativity", 0.0, 1.0, 0.3)
    if st.button("Save"):
        st.write(f"Saved, your name is {name}, your mood is {mood}, and your creativity is {creativity}")
prompt = st.chat_input("Say something. ")
if prompt:
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant"):
        st.write(f"You said: {prompt}")
