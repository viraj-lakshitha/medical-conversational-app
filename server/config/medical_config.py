"""
Medical System Configuration Management

Centralized configuration for all medical AI components including
NED pipeline, RAG system, Neuro-Symbolic model, and LLM integration.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class MedicalSystemConfig:
    """Centralized configuration for the medical conversational AI system"""
    
    # Base paths
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
    
    # NED Pipeline Configuration
    NED_CONFIG = {
        "ner_model_path": str(SCRIPTS_ROOT / "ned" / "biobert_ner_model"),
        "ranking_model_path": str(SCRIPTS_ROOT / "ned" / "neural_ranking_model.pth"),
        "confidence_threshold": 0.5,
        "max_candidates": 10
    }
    
    # RAG Pipeline Configuration  
    RAG_CONFIG = {
        "chromadb_host": os.getenv("CHROMA_HOST", "http://localhost:8000").replace("http://", "").replace("https://", "").split(":")[0],
        "chromadb_port": int(os.getenv("CHROMA_HOST", "http://localhost:8000").split(":")[-1]) if ":" in os.getenv("CHROMA_HOST", "http://localhost:8000") else 8000,
        "persist_directory": os.getenv("CHROMA_PERSIST_DIR", "./data/chroma"),
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "max_results": 5,
        "similarity_threshold": 0.0
    }
    
    # Neo4j Knowledge Base Configuration
    NEO4J_CONFIG = {
        "uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        "user": os.getenv("NEO4J_USER", "neo4j"),
        "password": os.getenv("NEO4J_PASSWORD", "password"),
        "database": os.getenv("NEO4J_DATABASE", "neo4j")
    }
    
    # Neuro-Symbolic Model Configuration
    NEURO_SYMBOLIC_CONFIG = {
        "model_path": str(SCRIPTS_ROOT / "neuro-symbolic" / "models" / "neo4j_disease_predictor.pkl"),
        "uncertainty_threshold": 0.25,
        "confidence_threshold": 0.75,
        "max_predictions": 5
    }
    
    # LLM Configuration (for future Qwen integration)
    LLM_CONFIG = {
        "model_path": str(SCRIPTS_ROOT / "llm_finetune" / "models" / "qwen-medical"),
        "max_length": 2048,
        "temperature": 0.7,
        "top_p": 0.9,
        "use_medical_reasoning": True
    }
    
    # Medical Pipeline Settings
    PIPELINE_CONFIG = {
        "enable_ned": True,
        "enable_rag": True,
        "enable_neuro_symbolic": True,
        "enable_llm_reasoning": False,  # Will enable when Qwen is trained
        "response_timeout": 30,  # seconds
        "cache_results": True,
        "log_medical_queries": True
    }
    
    # Safety and Validation Settings
    SAFETY_CONFIG = {
        "enable_medical_disclaimers": True,
        "minimum_confidence_for_diagnosis": 0.8,
        "require_human_review_threshold": 0.6,
        "enable_uncertainty_warnings": True,
        "max_disease_predictions": 3
    }
    
    @classmethod
    def validate_paths(cls):
        """Validate that all required model paths exist"""
        required_paths = [
            cls.NED_CONFIG["ner_model_path"],
            cls.NEURO_SYMBOLIC_CONFIG["model_path"]
        ]
        
        missing_paths = []
        for path in required_paths:
            if not os.path.exists(path):
                missing_paths.append(path)
        
        if missing_paths:
            print(f"Warning: Missing model files: {missing_paths}")
            return False
        return True
    
    @classmethod
    def get_connection_urls(cls):
        """Get formatted connection URLs for all services"""
        return {
            "neo4j": cls.NEO4J_CONFIG["uri"],
            "chromadb": f"http://{cls.RAG_CONFIG['chromadb_host']}:{cls.RAG_CONFIG['chromadb_port']}",
            "mongodb": os.getenv("MONGO_URI", "mongodb://localhost:27017")
        }