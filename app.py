# ===================================================================
# AI Research Agent - FULLY WORKING Streamlit Version )
# ===================================================================

import streamlit as st
from streamlit_chat import message
import os
import re
import json
import ast
import operator
import logging
import requests
import tempfile
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
from tqdm import tqdm
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
from groq import Groq
from gtts import gTTS

# Microphone Input
try:
    import speech_recognition as sr
    recognizer = sr.Recognizer()
    MIC_AVAILABLE = True
except ImportError:
    MIC_AVAILABLE = False

# ===================================================================
# ALL CLASSES - 100% COMPLETE & SYNTAX FIXED
# ===================================================================

class WebSearchTool:
    def __init__(self, max_results: int = 5, timeout: int = 10):
        self.max_results = max_results
        self.timeout = timeout
        self.base_url = "https://api.duckduckgo.com/"

    def search(self, query: str, num_results: Optional[int] = None) -> Dict[str, Any]:
        num_results = num_results or self.max_results
        try:
            params = {
                'q': query, 'format': 'json', 'no_redirect': '1',
                'no_html': '1', 'skip_disambig': '1'
            }
            response = requests.get(self.base_url, params=params, timeout=self.timeout,
                                  headers={'User-Agent': 'AI Research Agent 1.0'})
            response.raise_for_status()
            data = response.json()

            results = {
                'query': query,
                'abstract': data.get('Abstract', ''),
                'abstract_source': data.get('AbstractSource', ''),
                'answer': data.get('Answer', ''),
                'related_topics': [],
                'results_found': bool(any([data.get('Abstract'), data.get('Answer')]))
            }

            if 'RelatedTopics' in data:
                for topic in data['RelatedTopics'][:num_results]:
                    if isinstance(topic, dict) and 'Text' in topic:
                        results['related_topics'].append({
                            'text': topic.get('Text', ''),
                            'url': topic.get('FirstURL', '')
                        })
            return results
        except Exception as e:
            return {'query': query, 'error': str(e), 'results_found': False}

class DocumentProcessor:
    def __init__(self):
        self.supported_extensions = {'.txt', '.md', '.pdf'}

    def load_documents(self, data_directory: str) -> List[Dict[str, Any]]:
        documents = []
        data_path = Path(data_directory)
        if not data_path.exists():
            return documents

        files = [f for f in data_path.rglob('*') if f.suffix.lower() in self.supported_extensions]
        for file_path in tqdm(files, desc="Loading documents"):
            try:
                content = self._extract_text(file_path)
                if content.strip():
                    doc = {
                        'doc_id': str(file_path.relative_to(data_path)),
                        'content': content,
                        'file_path': str(file_path),
                        'file_type': file_path.suffix.lower()
                    }
                    documents.append(doc)
            except Exception as e:
                st.error(f"Error loading {file_path}: {e}")
        return documents

    def _extract_text(self, file_path: Path) -> str:
        extension = file_path.suffix.lower()
        if extension in ['.txt', '.md']:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif extension == '.pdf':
            text = ""
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text
        return ""

class DocumentChunker:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        chunks = []
        for doc in tqdm(documents, desc="Chunking documents"):
            doc_chunks = self._split_text(doc['content'])
            for i, chunk_text in enumerate(doc_chunks):
                chunk = {
                    'chunk_id': f"{doc['doc_id']}_chunk_{i}",
                    'content': chunk_text,
                    'doc_id': doc['doc_id'],
                    'chunk_index': i,
                    'source_file': doc['file_path'],
                    'file_type': doc['file_type']
                }
                chunks.append(chunk)
        return chunks

    def _split_text(self, text: str) -> List[str]:
        text = re.sub(r'\s+', ' ', text.strip())
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            if end >= len(text):
                chunks.append(text[start:])
                break

            chunk = text[start:end]
            last_sentence = max(chunk.rfind('.'), chunk.rfind('!'), chunk.rfind('?'))
            if last_sentence > start + self.chunk_size // 2:
                end = start + last_sentence + 1
            else:
                last_space = chunk.rfind(' ')
                if last_space > start + self.chunk_size // 2:
                    end = start + last_space

            chunks.append(text[start:end].strip())
            start = end - self.chunk_overlap

        return [chunk for chunk in chunks if len(chunk.strip()) > 10]

class EmbeddingGenerator:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, chunks: List[Dict[str, Any]]) -> np.ndarray:
        texts = [chunk['content'] for chunk in chunks]
        embeddings = self.model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
        return embeddings

    def get_query_embedding(self, query: str) -> np.ndarray:
        return self.model.encode([query], convert_to_numpy=True)[0]

def build_embeddings_from_directory(data_directory: str, chunk_size: int = 512, chunk_overlap: int = 50) -> Dict[str, Any]:
    os.makedirs("temp_embeddings", exist_ok=True)
    doc_processor = DocumentProcessor()
    chunker = DocumentChunker(chunk_size, chunk_overlap)
    embedder = EmbeddingGenerator()

    documents = doc_processor.load_documents(data_directory)
    if not documents:
        return {}

    chunks = chunker.chunk_documents(documents)
    embeddings = embedder.generate_embeddings(chunks)

    return {
        'chunks': chunks,
        'embeddings': embeddings,
        'metadata': {
            'num_documents': len(documents),
            'num_chunks': len(chunks),
            'embedding_dim': embeddings.shape[1]
        }
    }

class DocumentRetriever:
    def __init__(self, embedding_model_name: str = 'all-MiniLM-L6-v2'):
        self.embedding_generator = EmbeddingGenerator(embedding_model_name)
        self.index = None
        self.chunks = []
        self.embeddings = None

    def build_index(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray) -> None:
        self.chunks = chunks
        self.embeddings = embeddings
        embedding_dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(embedding_dim)
        embeddings_normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.index.add(embeddings_normalized.astype(np.float32))

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if not self.index:
            return []

        query_embedding = self.embedding_generator.get_query_embedding(query)
        query_normalized = query_embedding / np.linalg.norm(query_embedding)
        scores, indices = self.index.search(query_normalized.reshape(1, -1).astype(np.float32), k)

        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx >= 0:
                chunk = self.chunks[idx].copy()
                chunk.update({'similarity_score': float(score), 'rank': i + 1})
                results.append(chunk)
        return results

# ===================================================================
# AGENT TOOLS & PLANNER - FULL
# ===================================================================

class AgenticTools:
    def __init__(self):
        self.web_search_instance = WebSearchTool()

    def calculator_tool(self, expression: str) -> Dict[str, Any]:
        try:
            clean_expr = re.sub(r'[^0-9+\-*/().\s]', '', expression)
            node = ast.parse(clean_expr, mode='eval')
            result = self._eval_expr(node.body)
            return {"tool": "calculator", "result": result, "success": True}
        except Exception as e:
            return {"tool": "calculator", "success": False, "error": str(e)}

    def _eval_expr(self, node):
        ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}
        if isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.BinOp):
            return ops[type(node.op)](self._eval_expr(node.left), self._eval_expr(node.right))
        raise ValueError("Invalid")

    def web_search_tool(self, query: str) -> Dict[str, Any]:
        result = self.web_search_instance.search(query)
        return {"tool": "web_search", "result": result, "success": result.get('results_found', False)}

class AgentPlanner:
    def __init__(self):
        self.planning_patterns = {
            "calculation": ["calculate", "how much", "total"],
            "current_info": ["latest", "current", "price"],
            "analysis": ["analyze", "summary"]
        }

    def create_execution_plan(self, query: str) -> Dict[str, Any]:
        query_lower = query.lower()
        needed = []
        for cap, words in self.planning_patterns.items():
            if any(w in query_lower for w in words):
                needed.append(cap)

        steps = [{"step": 1, "tool": "document_search"}]
        if "calculation" in needed:
            steps.append({"step": 2, "tool": "calculator"})
        if "current_info" in needed:
            steps.append({"step": 2, "tool": "web_search"})
        steps.append({"step": 3, "tool": "synthesizer"})
        return {"steps": steps}

class ResultSynthesizer:
    def __init__(self, groq_client):
        self.groq_client = groq_client

    def synthesize_results(self, query: str, results: Dict[str, Any]) -> str:
        context = ""
        if results.get("document_search"):
            context += results["document_search"]
        if results.get("web_search"):
            context += str(results["web_search"].get("answer", ""))
        prompt = f"Question: {query}\nContext: {context}\nAnswer:"
        try:
            resp = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            return resp.choices[0].message.content
        except:
            return "Could not generate answer."

# ===================================================================
# MAIN AGENT CLASS
# ===================================================================

class AgenticRAGAgent:
    def __init__(self):
        self.retriever = DocumentRetriever()
        self.tools = AgenticTools()
        self.planner = AgentPlanner()
        self.groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
        self.synthesizer = ResultSynthesizer(self.groq_client)

    def clean_text_for_speech(self, text):
        return re.sub(r'[\*`_\[\]#]', '', text).strip()

    def generate_audio_response(self, text):
        clean = self.clean_text_for_speech(text)
        if not clean: return None
        try:
            path = f"/tmp/response_{int(time.time())}.mp3"
            gTTS(text=clean, lang='en').save(path)
            return path
        except:
            return None

    def process_query(self, query, history):
        history.append({"role": "user", "content": query})

        if "hi" in query.lower():
            resp = "Hi! I'm your AI Research Agent. Upload PDFs and ask anything!"
            audio = self.generate_audio_response(resp)
            history.append({"role": "assistant", "content": resp})
            return history, resp, audio

        if not self.retriever.index:
            resp = "Please upload and process a PDF first!"
            audio = self.generate_audio_response(resp)
            history.append({"role": "assistant", "content": resp})
            return history, resp, audio

        retrieved = self.retriever.search(query, k=5)
        context = "\n\n".join([d['content'] for d in retrieved])

        prompt = f"Question: {query}\nContext: {context}\nAnswer:"
        try:
            resp = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            ).choices[0].message.content
        except:
            resp = "Sorry, I couldn't answer."

        audio = self.generate_audio_response(resp)
        history.append({"role": "assistant", "content": resp})
        return history, resp, audio

    def upload_documents(self, files):
        os.makedirs("sample_data", exist_ok=True)
        for f in files:
            with open(f"sample_data/{f.name}", "wb") as out:
                out.write(f.getbuffer())
        data = build_embeddings_from_directory("sample_data")
        if data:
            self.retriever.build_index(data['chunks'], data['embeddings'])
            return "PDFs indexed successfully!"
        return "Failed to index PDFs"

# ===================================================================
# STREAMLIT UI
# ===================================================================

st.set_page_config(page_title="AI Research Agent", layout="wide")
st.markdown("<h1 style='text-align:center;color:#667eea;'>AI Research Agent</h1>", unsafe_allow_html=True)

if "agent" not in st.session_state:
    st.session_state.agent = AgenticRAGAgent()
    st.session_state.messages = []

agent = st.session_state.agent

with st.sidebar:
    st.header("PDF Upload")
    uploaded = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if st.button("Process Documents", type="primary"):
        if uploaded:
            with st.spinner("Processing..."):
                status = agent.upload_documents(uploaded)
                st.success(status)

for msg in st.session_state.messages:
    if msg["role"] == "user":
        message(msg["content"], is_user=True)
    else:
        message(msg["content"])

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
    message(user_query, is_user=True)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            history, response, audio_file = agent.process_query(user_query, st.session_state.messages.copy())
            st.session_state.messages = history
            st.write(response)
            if audio_file:
                st.audio(audio_file, autoplay=True)

if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()
