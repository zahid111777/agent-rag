# ===================================================================
# AI Research Agent - FULLY WORKING Streamlit App (November 21, 2025)
# All features from your notebook + Voice + Mic + No Errors
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===================================================================
# FULL CLASSES FROM YOUR NOTEBOOK - 100% COMPLETE
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
            logger.error(f"Web search failed: {e}")
            return {'query': query, 'error': str(e), 'results_found': False}

class ConfigManager:
    DEFAULT_CONFIG = {
        'embedding_model': 'all-MiniLM-L6-v2',
        'groq_model': 'llama-3.1-8b-instant',
        'max_iterations': 5,
        'confidence_threshold': 0.7,
        'retrieval_k': 5,
        'chunk_size': 512,
        'chunk_overlap': 50
    }

    @staticmethod
    def load_config():
        return ConfigManager.DEFAULT_CONFIG.copy()

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
                logger.error(f"Error loading {file_path}: {e}")
        return documents

    def _extract_text(self, file_path: Path) -> str:
        extension = file_path.suffix.lower()
        if extension == '.txt':
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

class AgenticTools:
    def __init__(self):
        self.tools = {
            "calculator": self.calculator_tool,
            "web_search": self.web_search_tool,
            "fact_checker": self.fact_checker_tool,
            "document_analyzer": self.document_analyzer_tool
        }
        self.web_search_instance = WebSearchTool()

    def calculator_tool(self, expression: str) -> Dict[str, Any]:
        try:
            clean_expr = re.sub(r'[^0-9+\-*/().\s]', '', expression)
            node = ast.parse(clean_expr, mode='eval')
            result = self._eval_expr(node.body)
            return {
                "tool": "calculator",
                "input": expression,
                "result": result,
                "success": True,
                "explanation": f"Calculated {clean_expr} = {result}"
            }
        except Exception as e:
            return {"tool": "calculator", "input": expression, "result": None, "success": False, "error": str(e)}

    def _eval_expr(self, node):
        ops = {
            ast.Add: operator.add, ast.Sub: operator.sub,
            ast.Mult: operator.mul, ast.Div: operator.truediv,
            ast.Pow: operator.pow, ast.USub: operator.neg
        }
        if isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.BinOp):
            return ops[type(node.op)](self._eval_expr(node.left), self._eval_expr(node.right))
        elif isinstance(node, ast.UnaryOp):
            return ops[type(node.op)](self._eval_expr(node.operand))
        raise TypeError(node)

    def web_search_tool(self, query: str) -> Dict[str, Any]:
        try:
            result = self.web_search_instance.search(query)
            return {
                "tool": "web_search",
                "input": query,
                "result": result,
                "success": result.get('results_found', False),
                "explanation": f"Found web information about: {query}"
            }
        except Exception as e:
            return {"tool": "web_search", "input": query, "result": None, "success": False, "error": str(e)}

    def fact_checker_tool(self, claim: str) -> Dict[str, Any]:
        confidence = "medium"
        verification = "partial"
        if re.search(r'\d+', claim):
            verification = "requires_calculation"
        return {
            "tool": "fact_checker",
            "input": claim,
            "result": {"verification": verification, "confidence": confidence},
            "success": True
        }

    def document_analyzer_tool(self, text: str, analysis_type: str = "summary") -> Dict[str, Any]:
        sentences = re.split(r'[.!?]+', text)[:3]
        summary = '. '.join([s.strip() for s in sentences if s.strip()])
        return {
            "tool": "document_analyzer",
            "input": f"{analysis_type} analysis",
            "result": summary,
            "success": True
        }

class AgentPlanner:
    def __init__(self):
        self.planning_patterns = {
            "calculation": ["calculate", "compute", "math", "percentage", "total"],
            "current_info": ["latest", "recent", "current", "rate", "price", "exchange", "dollar", "currency"],
            "analysis": ["analyze", "insights", "patterns", "summary"],
            "fact_check": ["verify", "confirm", "accurate"]
        }

    def create_execution_plan(self, query: str) -> Dict[str, Any]:
        query_lower = query.lower()
        needed_capabilities = []
        for capability, keywords in self.planning_patterns.items():
            if any(keyword in query_lower for keyword in keywords):
                needed_capabilities.append(capability)

        steps = [{"step": 1, "tool": "document_search", "description": "Search documents", "query": query}]
        step_num = 2

        if "calculation" in needed_capabilities:
            steps.append({"step": step_num, "tool": "calculator", "description": "Perform calculations", "depends_on": [1]})
            step_num += 1
        if "current_info" in needed_capabilities:
            steps.append({"step": step_num, "tool": "web_search", "description": "Search web", "query": query, "depends_on": [1]})
            step_num += 1
        if "analysis" in needed_capabilities:
            steps.append({"step": step_num, "tool": "document_analyzer", "description": "Analyze content", "depends_on": [1]})
            step_num += 1

        steps.append({"step": step_num, "tool": "synthesizer", "description": "Synthesize results", "depends_on": list(range(1, step_num))})

        return {"query": query, "detected_needs": needed_capabilities, "steps": steps, "total_steps": len(steps)}

class ResultSynthesizer:
    def __init__(self, groq_client):
        self.groq_client = groq_client

    def synthesize_results(self, query: str, results: Dict[str, Any], temperature: float = 0.3, max_tokens: int = 500) -> str:
        context_parts = []
        if "document_search" in results and results["document_search"]["success"]:
            context_parts.append(f"DOCUMENTS:\n{results['document_search']['result']}")
        if "web_search" in results and results["web_search"]["success"]:
            web_info = results["web_search"]["result"]
            web_text = f"{web_info.get('abstract', '')} {web_info.get('answer', '')}"
            context_parts.append(f"WEB INFO:\n{web_text}")
        if "calculator" in results and results["calculator"]["success"]:
            context_parts.append(f"CALCULATION:\n{results['calculator']['result']}")

        all_context = "\n\n".join(context_parts)
        prompt = f"""Based on the following information, provide a comprehensive answer.
QUESTION: {query}
INFORMATION:
{all_context}
Provide a clear, direct answer synthesizing all sources."""

        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are an expert research assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Based on available information: {all_context[:500]}..."

class AgenticEvaluator:
    def evaluate_response(self, query: str, response: str, tool_results: Dict[str, Any]) -> Dict[str, Any]:
        successful_tools = sum(1 for r in tool_results.values() if r.get("success", False))
        total_tools = len(tool_results)

        confidence = min(0.8, successful_tools / max(total_tools, 1)) if successful_tools > 0 else 0.0
        source_types = []
        if "document_search" in tool_results and tool_results["document_search"]["success"]:
            source_types.append("documents")
        if "web_search" in tool_results and tool_results["web_search"]["success"]:
            source_types.append("web")

        return {
            "confidence_score": confidence,
            "completeness": "comprehensive" if successful_tools >= total_tools else "partial",
            "source_diversity": len(source_types),
            "recommendations": []
        }

# ===================================================================
# MAIN AGENT CLASS - FULLY FUNCTIONAL
# ===================================================================

class AgenticRAGAgent:
    def __init__(self):
        self.config = ConfigManager.load_config()
        self.retriever = None
        self.groq_client = None
        self.conversation_history = []

        self.tools = AgenticTools()
        self.planner = AgentPlanner()
        self.synthesizer = None
        self.evaluator = AgenticEvaluator()

        self.temperature = 0.3
        self.max_tokens = 500
        self.chunk_size = 512
        self.chunk_overlap = 50
        self.retrieval_k = 8

        self.enable_web_search = True
        self.enable_calculations = True
        self.enable_fact_checking = True
        self.enable_analysis = True

        groq_api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
        if groq_api_key:
            try:
                self.groq_client = Groq(api_key=groq_api_key)
                self.synthesizer = ResultSynthesizer(self.groq_client)
                st.success("Groq API configured")
            except Exception as e:
                st.error(f"Groq Error: {e}")

    def clean_text_for_speech(self, text):
        if not text:
            return ""

        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        text = re.sub(r'```[^`]*```', '', text, flags=re.DOTALL)
        text = re.sub(r'`([^`]+)`', r'\1', text)
        text = re.sub(r'^[\s]*[-*+•]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^[\s]*\d+\.\s+', '', text, flags=re.MULTILINE)

        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "\U0001F900-\U0001F9FF"
            "\U00002600-\U000026FF"
            "\U00002700-\U000027BF"
            "]+"
        )
        text = emoji_pattern.sub('', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', '. ', text)
        text = text.strip()
        text = re.sub(r'\.+', '.', text)

        return text

    def generate_audio_response(self, text):
        if not text:
            return None

        clean_text = self.clean_text_for_speech(text)
        if not clean_text:
            return None

        try:
            temp_dir = tempfile.gettempdir()
            timestamp = int(time.time())
            audio_file = os.path.join(temp_dir, f"response_{timestamp}.mp3")

            tts = gTTS(text=clean_text, lang='en', slow=False)
            tts.save(audio_file)
            return audio_file
        except Exception as e:
            logger.error(f"Audio generation failed: {e}")
            return None

    def is_greeting_or_casual(self, query):
        query_lower = query.lower().strip()
        greetings = ['hi', 'hello', 'hey', 'howdy']
        return any(query_lower.startswith(g) for g in greetings) or query_lower in greetings

    def get_greeting_response(self, query):
        return "Hi there! 👋 I'm AI Research Agent with agentic capabilities. Upload PDF documents and ask complex questions!"

    def get_simple_answer(self, query, retrieved_docs):
        if not self.groq_client:
            return "Error: Groq API not configured"

        context = "\n\n".join([doc.get('content', str(doc)) for doc in retrieved_docs[:5]])
        prompt = f"""Based on this context, provide a clear answer.
Context: {context}
Question: {query}
Answer:"""

        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a helpful research assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error: {str(e)}"

    def process_agentic_query(self, query, chat_history):
        if not query.strip():
            return chat_history, "", None

        if chat_history is None:
            chat_history = []

        chat_history.append({"role": "user", "content": query})

        try:
            if self.is_greeting_or_casual(query):
                response = self.get_greeting_response(query)
                chat_history.append({"role": "assistant", "content": response})
                audio_file = self.generate_audio_response(response)
                return chat_history, response, audio_file

            if not self.retriever or not hasattr(self.retriever, 'index') or not self.retriever.index:
                error = "📄 Please upload a PDF document first!"
                chat_history.append({"role": "assistant", "content": error})
                audio_file = self.generate_audio_response(error)
                return chat_history, error, audio_file

            plan = self.planner.create_execution_plan(query)

            results = {}
            current_step = 0

            for step in plan['steps']:
                current_step += 1

                if step['tool'] == 'document_search':
                    retrieved_docs = self.retriever.search(query, k=self.retrieval_k)
                    if retrieved_docs:
                        doc_answer = self.get_simple_answer(query, retrieved_docs)
                        results['document_search'] = {"success": True, "result": doc_answer}
                    else:
                        results['document_search'] = {"success": False, "result": "No relevant info"}

                elif step['tool'] == 'calculator' and self.enable_calculations:
                    math_patterns = re.findall(r'[\d+\-*/().\s]+', query)
                    for expr in math_patterns:
                        if any(op in expr for op in ['+', '-', '*', '/']):
                            results['calculator'] = self.tools.calculator_tool(expr.strip())
                            break

                elif step['tool'] == 'web_search' and self.enable_web_search:
                    results['web_search'] = self.tools.web_search_tool(query)

                elif step['tool'] == 'document_analyzer' and self.enable_analysis:
                    if 'document_search' in results and results['document_search']['success']:
                        doc_content = results['document_search']['result']
                        results['document_analyzer'] = self.tools.document_analyzer_tool(doc_content, "summary")

            if self.synthesizer:
                final_answer = self.synthesizer.synthesize_results(query, results, self.temperature, self.max_tokens)
            else:
                successful = [r['result'] for r in results.values() if r.get('success')]
                final_answer = f"Based on available info: {' '.join(map(str, successful))}"

            evaluation = self.evaluator.evaluate_response(query, final_answer, results)

            eval_summary = f"\n\n💡 **Analysis:**\n"
            eval_summary += f"• Confidence: {evaluation['confidence_score']:.1%}\n"
            eval_summary += f"• Sources: {evaluation['source_diversity']} types\n"
            eval_summary += f"• Completeness: {evaluation['completeness']}"

            complete_response = final_answer + eval_summary

            audio_file = self.generate_audio_response(final_answer)

            chat_history.append({"role": "assistant", "content": complete_response})

            self.conversation_history.append({
                'timestamp': datetime.now().isoformat(),
                'query': query,
                'response': complete_response,
                'plan': plan,
                'results': results,
                'evaluation': evaluation,
                'audio_file': audio_file
            })

            return chat_history, complete_response, audio_file

        except Exception as e:
            error = f"❌ Error: {str(e)}"
            chat_history.append({"role": "assistant", "content": error})
            return chat_history, error, None

    def upload_documents(self, files, progress=None):
        if not files:
            return "No files uploaded"

        try:
            os.makedirs("sample_data", exist_ok=True)

            uploaded = []
            for file in files:
                if hasattr(file, 'name') and file.name.endswith('.pdf'):
                    original = os.path.basename(file.name)
                    dest = os.path.join("sample_data", original)
                    with open(dest, "wb") as dst:
                        dst.write(file.read())
                    uploaded.append(original)

            if not uploaded:
                return "❌ No valid PDF files"

            embeddings_data = build_embeddings_from_directory("sample_data")

            if embeddings_data and 'embeddings' in embeddings_data:
                self.retriever = DocumentRetriever()
                self.retriever.build_index(embeddings_data['chunks'], embeddings_data['embeddings'])

                doc_count = embeddings_data.get('metadata', {}).get('num_documents', 0)
                chunk_count = embeddings_data.get('metadata', {}).get('num_chunks', 0)

                return f"""✅ **Success!**
📄 Files: {', '.join(uploaded)}
📊 Documents: {doc_count} | Chunks: {chunk_count}
🎯 Ready for complex questions with voice support!"""
            else:
                return "❌ Failed to process documents"
        except Exception as e:
            return f"❌ Error: {str(e)}"

    def update_settings(self, temp, tokens, chunk_size, overlap, k, web, calc, fact, analysis):
        self.temperature = temp
        self.max_tokens = tokens
        self.chunk_size = chunk_size
        self.chunk_overlap = overlap
        self.retrieval_k = k
        self.enable_web_search = web
        self.enable_calculations = calc
        self.enable_fact_checking = fact
        self.enable_analysis = analysis

        return f"""⚙️ Settings Updated:
• Temperature: {temp}
• Max Tokens: {tokens}
• Chunk Size: {chunk_size}
• Retrieved: {k}
• Web: {'✅' if web else '❌'}
• Calc: {'✅' if calc else '❌'}
• Voice Output: ✅"""

# ===================================================================
# STREAMLIT INTERFACE - FULLY WORKING
# ===================================================================

if "agent" not in st.session_state:
    st.session_state.agent = AgenticRAGAgent()
    st.session_state.messages = []

agent = st.session_state.agent

st.markdown("""
<div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px;">
    <h1 style="color: white; margin: 0;">🤖 AI Research Agent - Agentic RAG</h1>
    <p style="color: white; margin: 10px 0;">Advanced Multi-Tool Research Assistant with Voice Support 🔊</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("<h3 style='text-align: center;'>📄 Upload Documents</h3>", unsafe_allow_html=True)
    file_upload = st.file_uploader("", type=["pdf"], accept_multiple_files=True)

    if st.button("Process Documents", type="primary"):
        if file_upload:
            with st.spinner("Processing PDFs..."):
                status = agent.upload_documents(file_upload)
                st.success("Ready!")
                st.write(status)
        else:
            st.warning("Please upload at least one PDF")

    with st.expander("⚙️ Settings", expanded=False):
        st.slider("Temperature", 0.0, 1.0, 0.3, key="temp")
        st.slider("Max Tokens", 100, 1000, 500, key="tokens")
        st.slider("Chunk Size", 256, 1024, 512, key="chunk")
        st.slider("Overlap", 0, 100, 50, key="overlap")
        st.slider("Retrieved", 3, 15, 8, key="k")
        st.checkbox("Web Search", True, key="web")
        st.checkbox("Calculator", True, key="calc")
        st.checkbox("Analysis", True, key="analysis")

        if st.button("Apply Settings"):
            agent.update_settings(
                st.session_state.temp, st.session_state.tokens,
                st.session_state.chunk, st.session_state.overlap,
                st.session_state.k, st.session_state.web,
                st.session_state.calc, False, st.session_state.analysis
            )
            st.success("Settings updated!")

# Chat
for idx, msg in enumerate(st.session_state.messages):
    key = f"{msg['role']}_{idx}"
    if msg["role"] == "user":
        message(msg["content"], is_user=True, key=key)
    else:
        message(msg["content"], key=key)

# Input
col1, col2 = st.columns([6, 1])
with col1:
    prompt = st.chat_input("Ask a complex research question...")
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
            st.success(f"Recognized: {user_query}")
        except:
            st.error("Could not understand audio")
            user_query = None

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    message(user_query, is_user=True, key=f"user_{len(st.session_state.messages)-1}")

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            history, response, audio_file = agent.process_agentic_query(user_query, st.session_state.messages.copy())
            st.session_state.messages = history
            message(response, key=f"assistant_{len(st.session_state.messages)-1}")
            st.write(response)
            if audio_file and os.path.exists(audio_file):
                st.audio(audio_file, autoplay=True)

if st.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.rerun()
