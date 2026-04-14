import streamlit as st
import speech_recognition as sr
from gtts import gTTS
import tempfile
import os

from preprocessing import (
    load_model_and_tokenizer,
    generate_medical_response,
    translate_text,
    detect_language,
    is_medical_query
)

st.set_page_config(page_title="Medical Voice AI", layout="wide")
st.title("🩺 Medical AI Assistant (Voice Enabled)")

@st.cache_resource
def load_model():
    return load_model_and_tokenizer()

tok, mdl, device = load_model()


if "messages" not in st.session_state:
    st.session_state.messages = []


with st.sidebar:
    st.header("⚙️ Settings")
    voice_mode = st.toggle("🎤 Voice Mode", value=False)

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


def listen_voice():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        st.info("🎤 Listening...")
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        return text
    except:
        return None

def speak_text(text):
    tts = gTTS(text)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        return fp.name

user_input = None

if voice_mode:
    if st.button("🎤 Speak"):
        user_input = listen_voice()
        if user_input:
            st.success(f"You said: {user_input}")
        else:
            st.error("Could not understand audio")
else:
    user_input = st.chat_input("Ask your medical question...")

if user_input:

    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            lang = detect_language(user_input)
            processed_input = translate_text(user_input, src=lang, dest="en")

            if not is_medical_query(processed_input):
                response = "❌ This doesn't seem like a medical query."
            else:
                prompt = (
        "You are a helpful and professional medical assistant. "
        "Provide a short, clear, and accurate answer to the user's question. "
        "If relevant, briefly mention key symptoms and treatments in 4 sentences. "
        "Focus only on health-related questions. " the query is not medical, respond politely asking for a medical question. "
        "Do not give emergency advice; recommend consulting a doctor if needed.\n"
        "Do not introduce yourself; go straight to the answer.\n"
        f"Question: {processed_input}\nAnswer:"
    )

                response = generate_medical_response(prompt, tok, mdl, device)
            st.markdown(response)

            audio_file = speak_text(response)
            st.audio(audio_file)

    st.session_state.messages.append({"role": "assistant", "content": response})


