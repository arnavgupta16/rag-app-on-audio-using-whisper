import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import streamlit as st
import openai
import os
import speech_recognition as sr


def get_audio_text(audio_files):
    r = sr.Recognizer()
    with sr.AudioFile(audio_files) as source:
        audio_data = r.record(source)
        text = r.recognize_whisper(audio_data)
        
    return text

def get_text_chunks(text, chunk_size=100, chunk_overlap=20):
    chunks = []
    start = 0
    end = chunk_size
    while end < len(text):
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
        end += chunk_size - chunk_overlap
    chunks.append(text[start:])
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def process_question(question, conversation_chain):
    response = conversation_chain.run(question)
    return response

def main():
  # Load environment variables (excluding OpenAI API key for now)

  st.set_page_config(page_title="Chat with multiple PDFs",
                    page_icon=":books:")

  # Get OpenAI API key from user input
  openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")

  # Set the OpenAI API key only if a value is entered
  if openai_api_key:
      os.environ["OPENAI_API_KEY"] = openai_api_key
      openai.api_key = openai_api_key

  if "conversation" not in st.session_state:
      st.session_state.conversation = None
  if "chat_history" not in st.session_state:
      st.session_state.chat_history = []

  st.header("Chat with multiple PDFs :books:")
  user_question = st.text_input("Ask your Question about your audio")
  with st.sidebar:
    st.subheader("Your audio")
    audio_files = st.file_uploader("Upload your audio", type=['wav'])

    if st.button("Process"):
        with st.spinner("Processing"):
            raw_text = get_audio_text(audio_files)
            st.write(f"Transcribed Text: \n{raw_text}")

            text_chunks = get_text_chunks(raw_text)
            vectorstore = get_vectorstore(text_chunks)
            st.session_state.conversation = get_conversation_chain(
                    vectorstore)

    if user_question:
        st.write(f"You: {user_question}")
        st.write(f"Bot: {process_question(user_question, st.session_state.conversation)}")

    if st.button("Transcribe Audio"):
        raw_text = get_audio_text(audio_files)
        st.session_state.raw_text = raw_text
        st.write(f"Transcribed Text: \n{raw_text}")
    
    if st.button("Download Transcribed Text"):
        if "raw_text" in st.session_state:
            raw_text = st.session_state.raw_text
            file_path = "transcribed_text.txt"
            with open(file_path, "w") as file:
                file.write(raw_text)
            st.success("Transcribed text downloaded successfully")

        # Download the file
            with open(file_path, "rb") as file:
                st.download_button(label='Click to download',
                               data=file,
                               file_name="transcribed_text.txt",
                               mime="text/plain")
        else:
            st.warning("Transcribed text is not available. Please transcribe the audio first.")
    
    if audio_files is not None:
        st.audio(audio_files)


if __name__ == '__main__':
    main()
