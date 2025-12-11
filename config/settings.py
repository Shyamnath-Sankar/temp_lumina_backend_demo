from pydantic_settings import BaseSettings
from typing import Optional, List
import os

class Settings(BaseSettings):
    SUPABASE_URL: str
    SUPABASE_KEY: str
    SUPABASE_SERVICE_KEY: str

    TOGETHER_API_KEY: str

    QDRANT_URL: str
    QDRANT_API_KEY: Optional[str] = None

    ENVIRONMENT: str = "production"
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8

    BACKEND_CORS_ORIGINS: List[str] = []

    UPLOAD_DIR: str = "/opt/render/project/src/uploads"
    MAX_FILE_SIZE: int = 10485760
    ALLOWED_EXTENSIONS: List[str] = ["pdf", "docx", "txt"]

    LLM_MODEL: str = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
    EMBEDDING_MODEL: str = "BAAI/bge-large-en-v1.5"
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    class Config:
        env_file = ".env"        # used only locally
        case_sensitive = True

settings = Settings()

# Ensure upload dir exists
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
