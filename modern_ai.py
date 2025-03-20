import streamlit as st
st.cache_resource.clear()
import google.generativeai as genai
import pyttsx3
import threading
import os
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from langchain.vectorstores import chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, TextLoader


load_dotenv()
API_KEY = os.getenv("MY_API_KEY")
genai.configure(api_key=API_KEY)

TEXT_MODEL_NAME = "gemini-1.5-pro-latest"
text_model = genai.GenerativeModel(TEXT_MODEL_NAME) 

from gtts import gTTS
import os

def speak(text):
    tts = gTTS(text=text, lang='en')
    tts.save("output.mp3")
    os.system("mpg321 output.mp3") 

speak("Hello, welcome!")


st.set_page_config(page_title="Modern AI", page_icon="🤖")
st.title("🤖 I am your assistant! How can I help you? 😊")

if "messages" not in st.session_state:
    st.session_state.messages = []

def generate_text_response(prompt: str) -> str:
    """Generate text response using Gemini API."""
    response = text_model.generate_content(prompt)
    return response.text if response and response.text else "I'm not sure what to say, but I'm here to help! 😊"

st.sidebar.title("⚙️ Settings")
st.sidebar.subheader("📜 Chat History")
if st.session_state.messages:
    for idx, msg in enumerate(st.session_state.messages):
        if msg["type"] == "user":
            st.sidebar.button(f"{idx+1}. {msg['content'][:30]}...", key=f"history_{idx}")

st.sidebar.subheader("⚡ Quick Prompts")
if st.sidebar.button("Tell me a fun fact! 🤩"):
    user_input = "Tell me a fun fact"
if st.sidebar.button("Give me a motivational quote 💪"):
    user_input = "Give me a motivational quote"
if st.sidebar.button("Summarize this topic for me 📚"):
    user_input = "Summarize this topic"

st.sidebar.subheader("🗂️ Upload a Document for AI Assistance")
doc_file = st.sidebar.file_uploader("📂 Upload a TXT or PDF", type=["txt", "pdf"])
if doc_file is not None:
    file_path = f"temp_{doc_file.name}"
    with open(file_path, "wb") as f:
        f.write(doc_file.read())
    
    if doc_file.type == "application/pdf":
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)
    
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    vector_db = FAISS.from_documents(texts, embeddings)
    st.session_state.vector_db = vector_db
    st.sidebar.success("✅ Document uploaded and processed!")

st.sidebar.subheader("📊 Upload a CSV for Data Analysis")
csv_file = st.sidebar.file_uploader("📂 Upload a CSV", type=["csv"])
if csv_file is not None:
    df = pd.read_csv(csv_file)
    st.sidebar.write("### 📈 Data Preview:", df.head())
    st.sidebar.write("### 📊 Basic Statistics:", df.describe())
    
    st.sidebar.subheader("📌 Column Analysis")
    column = st.sidebar.selectbox("Select a column", df.columns)
    if column:
        st.sidebar.write(f"### 📊 Distribution of {column}")
        fig, ax = plt.subplots()
        df[column].hist(ax=ax, bins=20, edgecolor='black')
        st.sidebar.pyplot(fig)

for msg in st.session_state.messages:
    if msg["type"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    elif msg["type"] == "text":
        st.markdown(f"**Modern AI:** {msg['content']}")
        engine.say(msg['content'])
        engine.runAndWait()

def on_input_change():
    user_input = st.session_state.user_input.strip()
    if user_input:
        st.session_state.messages.append({"type": "user", "content": user_input})
        
        # Check if knowledge base is available
        if "vector_db" in st.session_state:
            vector_db = st.session_state.vector_db
            response = vector_db.similarity_search(user_input, k=1)
            ai_response = response[0].page_content if response else "I couldn't find relevant information in the document."
        else:
            ai_response = generate_text_response(user_input)
        
        st.session_state.messages.append({"type": "text", "content": ai_response})
        st.session_state.user_input = ""

st.text_input("💬 Type your message", key="user_input", on_change=on_input_change)

st.subheader("✨ Quick Replies")
quick_replies = ["Tell me a joke", "Give me a fun fact", "Inspire me", "What's the latest tech trend?"]
for reply in quick_replies:
    if st.button(reply):
        st.session_state.messages.append({"type": "user", "content": reply})
        ai_response = generate_text_response(reply)
        st.session_state.messages.append({"type": "text", "content": ai_response})

st.write("Streamlit version:", st.__version__)
