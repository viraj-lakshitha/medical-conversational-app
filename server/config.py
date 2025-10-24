import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv(dotenv_path=".env")

# MongoDB Configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "app")

# ChromaDB Configuration
CHROMA_HOST = os.getenv("CHROMA_HOST", "http://localhost:8000")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")

# Security Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "super-secret-key")
API_PORT = int(os.getenv("API_PORT", 8080))