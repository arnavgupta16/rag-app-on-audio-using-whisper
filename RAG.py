import os
import openai
import speech_recognition as sr
from dotenv import load_dotenv
import base64
import streamlit as st
# Load your OpenAI key
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

st.set_page_config(page_title="Chat with audio AI")
st.header("Ask Your audio📄")
audio_file = st.file_uploader("Upload your audio", type=['wav'])

def transcribe_audio(audio_file):
    # Use the default SpeechRecognition recognizer
    r = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = r.record(source)
        text = r.recognize_whisper(audio_data)
    return text

def text_to_download_link(text, filename, title):
    b64 = base64.b64encode(text.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{filename}">{title}</a>'

if audio_file is not None:
    # Show audio player
    st.audio(audio_file, format='audio/wav')

    # Add transcribe button
    if st.button('Transcribe'):
        transcription = transcribe_audio((audio_file))
        st.write(transcription)
        st.markdown(text_to_download_link(transcription, "transcription.txt", "Download transcription"), unsafe_allow_html=True)

# Ask a question after the transcription
query = st.text_input("Ask your Question about your audio")
if st.button('Submit Query'):
    with open('transcription.txt', 'r') as f:
        transcription = f.read()

    # RAG Model
    from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration

    tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
    retriever = RagRetriever.from_pretrained("facebook/rag-token-nq", index_name="exact", use_dummy_dataset=True)
    model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)

    inputs = tokenizer.prepare_seq2seq_batch(["Question: " + query], return_tensors="pt")
    generated = model.generate(**inputs)

    response = tokenizer.batch_decode(generated, skip_special_tokens=True)
    st.success(response[0])
