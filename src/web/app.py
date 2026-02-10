"""
Web interface for RAG Knowledge Base.
FastAPI backend serving a glassmorphism UI.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import logging
import shutil
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.rag_chain import RAGChain
from src.config import settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Knowledge Base", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG chain
rag: Optional[RAGChain] = None


def get_rag() -> RAGChain:
    global rag
    if rag is None:
        rag = RAGChain()
    return rag


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve the glassmorphism UI."""
    return HTML_TEMPLATE


@app.post("/api/query")
async def query(request: QueryRequest):
    """Query the knowledge base."""
    try:
        chain = get_rag()
        result = chain.query(request.question, top_k=request.top_k)
        return result
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a document to the knowledge base."""
    allowed_extensions = {".pdf", ".txt", ".md", ".docx"}
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Allowed: {allowed_extensions}",
        )

    # Save uploaded file
    upload_dir = Path(settings.raw_data_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / file.filename

    try:
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Add to knowledge base
        chain = get_rag()
        count = chain.add_documents(str(file_path))

        return {
            "filename": file.filename,
            "chunks_added": count,
            "message": f"Successfully added {count} chunks from {file.filename}",
        }
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats")
async def stats():
    """Get knowledge base statistics."""
    try:
        chain = get_rag()
        return chain.get_stats()
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/clear")
async def clear():
    """Clear the knowledge base."""
    try:
        chain = get_rag()
        chain.retriever.vector_store.clear_collection()
        return {"message": "Knowledge base cleared successfully"}
    except Exception as e:
        logger.error(f"Clear error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG Knowledge Base</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', sans-serif;
            min-height: 100vh;
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
            color: #e0e0e0;
            overflow-x: hidden;
        }

        /* Animated background blobs */
        .bg-blob {
            position: fixed;
            border-radius: 50%;
            filter: blur(80px);
            opacity: 0.4;
            z-index: 0;
            animation: float 20s infinite ease-in-out;
        }

        .bg-blob:nth-child(1) {
            width: 500px; height: 500px;
            background: #7f5af0;
            top: -100px; left: -100px;
        }

        .bg-blob:nth-child(2) {
            width: 400px; height: 400px;
            background: #2cb67d;
            bottom: -50px; right: -50px;
            animation-delay: -7s;
        }

        .bg-blob:nth-child(3) {
            width: 300px; height: 300px;
            background: #e16162;
            top: 50%; left: 50%;
            animation-delay: -14s;
        }

        @keyframes float {
            0%, 100% { transform: translate(0, 0) scale(1); }
            25% { transform: translate(50px, -30px) scale(1.05); }
            50% { transform: translate(-20px, 40px) scale(0.95); }
            75% { transform: translate(30px, 20px) scale(1.02); }
        }

        /* Glass card */
        .glass {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
        }

        .container {
            max-width: 900px;
            margin: 0 auto;
            padding: 40px 20px;
            position: relative;
            z-index: 1;
        }

        /* Header */
        .header {
            text-align: center;
            margin-bottom: 40px;
        }

        .header h1 {
            font-size: 2.4rem;
            font-weight: 700;
            background: linear-gradient(135deg, #7f5af0, #2cb67d);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 8px;
        }

        .header p {
            font-size: 1rem;
            color: #94a1b2;
            font-weight: 300;
        }

        /* Stats bar */
        .stats-bar {
            display: flex;
            justify-content: center;
            gap: 24px;
            margin-bottom: 32px;
        }

        .stat-item {
            padding: 12px 24px;
            text-align: center;
        }

        .stat-item .stat-value {
            font-size: 1.5rem;
            font-weight: 600;
            color: #7f5af0;
        }

        .stat-item .stat-label {
            font-size: 0.75rem;
            color: #94a1b2;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        /* Upload section */
        .upload-section {
            padding: 24px;
            margin-bottom: 24px;
        }

        .upload-zone {
            border: 2px dashed rgba(127, 90, 240, 0.3);
            border-radius: 12px;
            padding: 32px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .upload-zone:hover {
            border-color: #7f5af0;
            background: rgba(127, 90, 240, 0.05);
        }

        .upload-zone.dragover {
            border-color: #2cb67d;
            background: rgba(44, 182, 125, 0.1);
        }

        .upload-icon {
            font-size: 2rem;
            margin-bottom: 8px;
        }

        .upload-zone p {
            color: #94a1b2;
            font-size: 0.9rem;
        }

        .upload-zone .formats {
            font-size: 0.75rem;
            color: #72757e;
            margin-top: 8px;
        }

        #file-input {
            display: none;
        }

        /* Query section */
        .query-section {
            padding: 24px;
            margin-bottom: 24px;
        }

        .input-group {
            display: flex;
            gap: 12px;
        }

        .input-group input {
            flex: 1;
            padding: 14px 20px;
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            background: rgba(255, 255, 255, 0.05);
            color: #e0e0e0;
            font-size: 1rem;
            font-family: 'Inter', sans-serif;
            outline: none;
            transition: border-color 0.3s;
        }

        .input-group input:focus {
            border-color: #7f5af0;
        }

        .input-group input::placeholder {
            color: #72757e;
        }

        .btn {
            padding: 14px 28px;
            border: none;
            border-radius: 12px;
            cursor: pointer;
            font-size: 0.95rem;
            font-weight: 500;
            font-family: 'Inter', sans-serif;
            transition: all 0.3s ease;
        }

        .btn-primary {
            background: linear-gradient(135deg, #7f5af0, #6246d4);
            color: white;
        }

        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(127, 90, 240, 0.35);
        }

        .btn-primary:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }

        .btn-danger {
            background: rgba(225, 97, 98, 0.15);
            color: #e16162;
            border: 1px solid rgba(225, 97, 98, 0.2);
            padding: 8px 16px;
            font-size: 0.8rem;
        }

        .btn-danger:hover {
            background: rgba(225, 97, 98, 0.25);
        }

        /* Results */
        .result-section {
            padding: 24px;
            margin-bottom: 24px;
            display: none;
        }

        .result-section.visible {
            display: block;
        }

        .result-header {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 16px;
            font-size: 0.85rem;
            color: #94a1b2;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .answer-text {
            font-size: 1rem;
            line-height: 1.7;
            color: #e0e0e0;
            white-space: pre-wrap;
        }

        .sources {
            margin-top: 16px;
            padding-top: 16px;
            border-top: 1px solid rgba(255, 255, 255, 0.06);
        }

        .sources-label {
            font-size: 0.75rem;
            color: #72757e;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
        }

        .source-tag {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            background: rgba(44, 182, 125, 0.1);
            color: #2cb67d;
            font-size: 0.8rem;
            margin: 4px 4px 4px 0;
        }

        /* Loading */
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }

        .loading.visible {
            display: block;
        }

        .spinner {
            width: 32px;
            height: 32px;
            border: 3px solid rgba(127, 90, 240, 0.2);
            border-top-color: #7f5af0;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
            margin: 0 auto 12px;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        .loading p {
            color: #94a1b2;
            font-size: 0.9rem;
        }

        /* Toast notifications */
        .toast {
            position: fixed;
            bottom: 24px;
            right: 24px;
            padding: 14px 24px;
            border-radius: 12px;
            font-size: 0.9rem;
            z-index: 100;
            transform: translateY(100px);
            opacity: 0;
            transition: all 0.3s ease;
        }

        .toast.show {
            transform: translateY(0);
            opacity: 1;
        }

        .toast-success {
            background: rgba(44, 182, 125, 0.15);
            border: 1px solid rgba(44, 182, 125, 0.3);
            color: #2cb67d;
        }

        .toast-error {
            background: rgba(225, 97, 98, 0.15);
            border: 1px solid rgba(225, 97, 98, 0.3);
            color: #e16162;
        }

        /* Footer */
        .footer {
            text-align: center;
            margin-top: 24px;
            display: flex;
            justify-content: center;
            gap: 16px;
            align-items: center;
        }

        /* Responsive */
        @media (max-width: 640px) {
            .header h1 { font-size: 1.8rem; }
            .input-group { flex-direction: column; }
            .stats-bar { flex-direction: column; align-items: center; }
        }
    </style>
</head>
<body>
    <div class="bg-blob"></div>
    <div class="bg-blob"></div>
    <div class="bg-blob"></div>

    <div class="container">
        <div class="header">
            <h1>RAG Knowledge Base</h1>
            <p>Upload documents and ask questions against your knowledge base</p>
        </div>

        <div class="stats-bar glass">
            <div class="stat-item">
                <div class="stat-value" id="doc-count">--</div>
                <div class="stat-label">Documents</div>
            </div>
            <div class="stat-item">
                <div class="stat-value" id="collection-name">--</div>
                <div class="stat-label">Collection</div>
            </div>
        </div>

        <div class="upload-section glass">
            <div class="upload-zone" id="upload-zone">
                <div class="upload-icon">+</div>
                <p>Drop files here or click to upload</p>
                <p class="formats">PDF, TXT, DOCX, MD</p>
            </div>
            <input type="file" id="file-input" accept=".pdf,.txt,.md,.docx">
        </div>

        <div class="query-section glass">
            <div class="input-group">
                <input
                    type="text"
                    id="query-input"
                    placeholder="Ask a question about your documents..."
                    autocomplete="off"
                >
                <button class="btn btn-primary" id="ask-btn" onclick="askQuestion()">Ask</button>
            </div>
        </div>

        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Searching knowledge base...</p>
        </div>

        <div class="result-section glass" id="result-section">
            <div class="result-header">Answer</div>
            <div class="answer-text" id="answer-text"></div>
            <div class="sources" id="sources-section">
                <div class="sources-label">Sources</div>
                <div id="sources-list"></div>
            </div>
        </div>

        <div class="footer">
            <button class="btn btn-danger" onclick="clearKnowledgeBase()">Clear Knowledge Base</button>
        </div>
    </div>

    <div class="toast" id="toast"></div>

    <script>
        // Load stats on page load
        document.addEventListener('DOMContentLoaded', loadStats);

        // Enter key to submit
        document.getElementById('query-input').addEventListener('keydown', function(e) {
            if (e.key === 'Enter') askQuestion();
        });

        // File upload handlers
        const uploadZone = document.getElementById('upload-zone');
        const fileInput = document.getElementById('file-input');

        uploadZone.addEventListener('click', () => fileInput.click());

        uploadZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadZone.classList.add('dragover');
        });

        uploadZone.addEventListener('dragleave', () => {
            uploadZone.classList.remove('dragover');
        });

        uploadZone.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadZone.classList.remove('dragover');
            if (e.dataTransfer.files.length) {
                uploadFile(e.dataTransfer.files[0]);
            }
        });

        fileInput.addEventListener('change', () => {
            if (fileInput.files.length) {
                uploadFile(fileInput.files[0]);
            }
        });

        async function loadStats() {
            try {
                const res = await fetch('/api/stats');
                const data = await res.json();
                document.getElementById('doc-count').textContent = data.document_count || 0;
                document.getElementById('collection-name').textContent = data.collection_name || '--';
            } catch (e) {
                console.error('Failed to load stats:', e);
            }
        }

        async function uploadFile(file) {
            const formData = new FormData();
            formData.append('file', file);

            uploadZone.innerHTML = '<div class="upload-icon">...</div><p>Uploading ' + file.name + '</p>';

            try {
                const res = await fetch('/api/upload', { method: 'POST', body: formData });
                const data = await res.json();

                if (!res.ok) throw new Error(data.detail || 'Upload failed');

                showToast(data.message, 'success');
                loadStats();
            } catch (e) {
                showToast('Upload failed: ' + e.message, 'error');
            } finally {
                uploadZone.innerHTML = '<div class="upload-icon">+</div><p>Drop files here or click to upload</p><p class="formats">PDF, TXT, DOCX, MD</p>';
                fileInput.value = '';
            }
        }

        async function askQuestion() {
            const input = document.getElementById('query-input');
            const question = input.value.trim();
            if (!question) return;

            const btn = document.getElementById('ask-btn');
            const loading = document.getElementById('loading');
            const resultSection = document.getElementById('result-section');

            btn.disabled = true;
            loading.classList.add('visible');
            resultSection.classList.remove('visible');

            try {
                const res = await fetch('/api/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question: question })
                });
                const data = await res.json();

                if (!res.ok) throw new Error(data.detail || 'Query failed');

                document.getElementById('answer-text').textContent = data.answer;

                const sourcesList = document.getElementById('sources-list');
                sourcesList.innerHTML = '';
                if (data.sources && data.sources.length > 0) {
                    data.sources.forEach(src => {
                        const tag = document.createElement('span');
                        tag.className = 'source-tag';
                        tag.textContent = src;
                        sourcesList.appendChild(tag);
                    });
                    document.getElementById('sources-section').style.display = 'block';
                } else {
                    document.getElementById('sources-section').style.display = 'none';
                }

                resultSection.classList.add('visible');
            } catch (e) {
                showToast('Error: ' + e.message, 'error');
            } finally {
                btn.disabled = false;
                loading.classList.remove('visible');
            }
        }

        async function clearKnowledgeBase() {
            if (!confirm('Are you sure you want to clear all documents?')) return;

            try {
                const res = await fetch('/api/clear', { method: 'POST' });
                const data = await res.json();

                if (!res.ok) throw new Error(data.detail || 'Clear failed');

                showToast('Knowledge base cleared', 'success');
                loadStats();
                document.getElementById('result-section').classList.remove('visible');
            } catch (e) {
                showToast('Error: ' + e.message, 'error');
            }
        }

        function showToast(message, type) {
            const toast = document.getElementById('toast');
            toast.textContent = message;
            toast.className = 'toast toast-' + type + ' show';
            setTimeout(() => toast.classList.remove('show'), 4000);
        }
    </script>
</body>
</html>"""
