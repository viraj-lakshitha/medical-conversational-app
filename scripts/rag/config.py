"""
Configuration settings for the RAG system
"""
import os
from typing import Dict, Any


class RAGConfig:
    CHROMADB_HOST = os.getenv("CHROMADB_HOST", "localhost")
    CHROMADB_PORT = int(os.getenv("CHROMADB_PORT", "8000"))

    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    COLLECTIONS = {
        "clinical_notes": "clinical_notes",
        "disease_info": "disease_info",
        "patient_records": "patient_records"
    }

    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

    MAX_RESULTS = int(os.getenv("MAX_RESULTS", "5"))
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.0"))

    DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

    @classmethod


    def get_chromadb_url(cls) -> str:
        return f"http://{cls.CHROMADB_HOST}:{cls.CHROMADB_PORT}"
