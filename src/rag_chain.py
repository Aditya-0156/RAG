"""
RAG Chain for Knowledge Base.
Combines retrieval with LLM generation to answer questions.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from typing import Optional, Dict, Any
import logging

from google import genai
from google.genai import types

from src.config import settings
from src.retrieval.retriever import Retriever
from src.ingestion.document_loader import DocumentLoader

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.

INSTRUCTIONS:
1. Answer the question using ONLY the information from the context provided.
2. If the context doesn't contain enough information, say "I don't have enough information to answer this question based on the available documents."
3. Be concise and direct in your answers.
4. If relevant, mention which document/source the information came from.
5. Do not make up information that isn't in the context.

CONTEXT:
{context}

---

USER QUESTION: {question}

ANSWER:"""


class RAGChain:
    """
    RAG (Retrieval Augmented Generation) Chain.

    Combines document retrieval with LLM generation to provide
    accurate answers based on your knowledge base.
    """

    def __init__(self, collection_name: Optional[str] = None):
        """
        Initialize the RAG chain.

        Args:
            collection_name: Name of the vector store collection.
        """
        if not settings.google_api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment")

        self.client = genai.Client(api_key=settings.google_api_key)
        self.model_name = settings.llm_model
        self.retriever = Retriever(collection_name=collection_name)

        logger.info(f"RAG Chain initialized with model: {settings.llm_model}")

    def query(self, question: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        Answer a question using RAG.

        Args:
            question: User's question.
            top_k: Number of documents to retrieve.

        Returns:
            Dict with 'answer', 'sources', and 'context'.
        """
        top_k = top_k or settings.top_k_results

        logger.info(f"Processing question: '{question[:50]}...'")

        # Retrieve relevant documents
        documents = self.retriever.retrieve(question, top_k=top_k)

        if not documents:
            return {
                "answer": (
                    "I couldn't find any relevant documents to answer your "
                    "question. Please make sure documents have been added "
                    "to the knowledge base."
                ),
                "sources": [],
                "context": "",
            }

        # Format context and build prompt
        context = self.retriever.format_context(documents)
        prompt = SYSTEM_PROMPT.format(context=context, question=question)

        # Generate answer with LLM
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=settings.temperature,
                ),
            )
            answer = response.text
        except Exception as e:
            logger.error(f"LLM error: {e}")
            answer = f"Error generating answer: {e}"

        # Extract sources
        sources = list(set(
            doc.metadata.get("source", "Unknown")
            for doc in documents
        ))

        logger.info("Answer generated successfully")

        return {
            "answer": answer,
            "sources": sources,
            "context": context,
        }

    def add_documents(self, file_path: str) -> int:
        """
        Add documents from a file to the knowledge base.

        Args:
            file_path: Path to file or directory.

        Returns:
            Number of chunks added.
        """
        loader = DocumentLoader()

        path = Path(file_path)
        if path.is_file():
            chunks = loader.load_document(file_path)
        elif path.is_dir():
            chunks = loader.load_directory(file_path)
        else:
            raise ValueError(f"Invalid path: {file_path}")

        # Add to vector store
        texts = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        added = self.retriever.vector_store.add_texts(texts, metadatas)

        logger.info(f"Added {added} document chunks to knowledge base")

        return added

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base."""
        return self.retriever.vector_store.get_collection_stats()
