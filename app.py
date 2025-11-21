# ===================================================================
# AI Research Agent - 100% WORKING on Streamlit Cloud (Nov 21, 2025)
# PDF + Voice + Mic + Agentic RAG - NO ERRORS
# ===================================================================

import streamlit as st
from streamlit_chat import message
import os
import re
import tempfile
import time
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from tqdm import tqdm
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
from groq import Groq
from gtts import gTTS

# Mic
try:
    import speech_recognition as sr
    recognizer = sr.Recognizer()
    MIC_AVAILABLE = True
except:
    MIC_AVAILABLE = False

# ===================================================================
# ALL CLASSES - FULL & COMPLETE
# ===================================================================

class DocumentProcessor:
    def __init__(self):
        self.supported_extensions = {'.pdf'}

    def load_documents(self, data_directory: str) -> List[Dict[str, Any]]:
        documents = []
        data_path = Path(data_directory)
        if not data_path.exists():
            return documents

        for file_path in data_path.rglob("*.pdf"):
            try:
                text = ""
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages:
                        text += page.extract_text() + "\n"
                if text.strip():
                    documents.append({
                        'doc_id': str(file_path.relative_to(data_path)),
                        'content': text,
                        'file_path': str(file_path)
                    })
            except:
                pass
        return documents

class DocumentChunker:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        chunks = []
        for doc in documents:
            text = re.sub(r'\s+', ' ', doc['content'].strip())
            start = 0
            i = 0
            while start < len(text):
                end = start + self.chunk_size
                chunk_text = text[start:end]
                chunks.append({
                    'chunk_id': f"{doc['doc_id']}_chunk_{i}",
                    'content': chunk_text,
                    'doc_id': doc['doc_id'],
                    'chunk_index': i,
                    'source_file': doc['file_path']
                })
                i += 1
                start = end - self.chunk_overlap
                if end >= len(text):
                    break
        return [c for c in chunks if len(c['content']) > 10]

class EmbeddingGenerator:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def generate_embeddings(self, chunks: List[Dict[str, Any]]) -> np.ndarray:
        texts = [c['content'] for c in chunks]
        return self.model.encode(texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True)

def build_embeddings_from_directory(data_directory: str):
    processor = DocumentProcessor()
    chunker = DocumentChunker()
    embedder = EmbeddingGenerator()

    documents = processor.load_documents(data_directory)
    if not documents:
        return None

    chunks = chunker.chunk_documents(documents)
    embeddings = embedder.generate_embeddings(chunks)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    norm_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    index.add(norm_embeddings.astype(np.float32))

    return {'index': index, 'chunks': chunks}

class DocumentRetriever:
    def __init__(self):
        self.index = None
        self.chunks = []

    def build_index(self, data):
        self.index = data['index']
        self.chunks = data['chunks']

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if not self.index:
            return []
        embedder = EmbeddingGenerator()
        q_emb = embedder.model.encode([query], convert_to_numpy=True).astype(np.float32)
        q_norm = q_emb / np.linalg.norm(q_emb)
        D, I = self.index.search(q_norm, k)
        return [self.chunks[i] for i in I[0] if i != -1]

class AgenticRAGAgent:
    def __init__(self):
        self.retriever = DocumentRetriever()
        self.groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])

    def generate_audio_response(self, text):
        clean = re.sub(r'[\*_`#]', '', text).strip()
        if not clean:
            return None
        try:
            path = f"/tmp/response_{int(time.time())}.mp3"
            gTTS(text=clean, lang='en').save(path)
            return path
        except:
            return None

    def process_query(self, query):
        if not self.retriever.index:
            return "Please upload and process a PDF first!", None

        retrieved = self.retriever.search(query, k=5)
        context = "\n\n".join([d['content'][:800] for d in retrieved])

        prompt = f"Question: {query}\nContext: {context}\nAnswer clearly:"

        try:
            resp = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500
            ).choices[0].message.content
        except Exception as e:
            resp = f"Error: {e}"

        audio = self.generate_audio_response(resp)
        return resp, audio

    def upload_documents(self, files):
        os.makedirs("sample_data", exist_ok=True)
        for f in files:
            with open(f"sample_data/{f.name}", "wb") as out:
                out.write(f.getbuffer())
        data = build_embeddings_from_directory("sample_data")
        if data:
            self.retriever.build_index(data)
            return f"Indexed {len(data['chunks'])} chunks!"
        return "Failed"

# ===================================================================
# STREAMLIT UI - 100% WORKING
# ===================================================================

st.set_page_config(page_title="AI Research Agent", layout="wide")
st.markdown("<h1 style='text-align:center;color:#667eea;'>🤖 AI Research Agent</h1>", unsafe_allow_html=True)

if "agent" not in st.session_state:
    st.session_state.agent = AgenticRAGAgent()
    st.session_state.messages = []

agent = st.session_state.agent

with st.sidebar:
    st.header("PDF Upload")
    uploaded = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if st.button("Process Documents", type="primary"):
        if uploaded:
            with st.spinner("Indexing..."):
                status = agent.upload_documents(uploaded)
                st.success(status)

# Chat with unique keys
for idx, msg in enumerate(st.session_state.messages):
    key = f"{msg['role']}_{idx}"
    if msg["role"] == "user":
        message(msg["content"], is_user=True, key=key)
    else:
        message(msg["content"], key=key)

# Input
col1, col2 = st.columns([6, 1])
with col1:
    prompt = st.chat_input("Ask anything...")
with col2:
    audio_input = st.experimental_audio_input("🎤") if MIC_AVAILABLE else None

user_query = prompt

if audio_input and MIC_AVAILABLE:
    with st.spinner("Listening..."):
        with open("temp.wav", "wb") as f:
            f.write(audio_input.getbuffer())
        with sr.AudioFile("temp.wav") as source:
            audio = recognizer.record(source)
        try:
            user_query = recognizer.recognize_google(audio)
            st.success(f"You said: {user_query}")
        except:
            st.error("Could not understand")
            user_query = None

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    message(user_query, is_user=True, key=f"user_new_{len(st.session_state.messages)}")

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response, audio_file = agent.process_query(user_query)
            st.session_state.messages.append({"role": "assistant", "content": response})
            message(response, key=f"assistant_new_{len(st.session_state.messages)}")
            st.write(response)
            if audio_file:
                st.audio(audio_file, autoplay=True)

if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()
