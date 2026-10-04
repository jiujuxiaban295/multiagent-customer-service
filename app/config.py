"""Configuration is loaded once at application startup, never from model arguments."""
from dataclasses import dataclass
from pathlib import Path
import os
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Settings:
    model: str = 'deepseek-v4-flash'
    model_base_url: str = 'https://api.deepseek.com/anthropic'
    model_api_key: str = ''
    jev_api_key: str = ''
    jev_model: str = 'jev-1.13.0'
    jev_endpoint: str = 'https://api.typesafe.ai/v1/systemone'
    jev_timeout: float = 8.0
    redis_url: str = 'redis://127.0.0.1:6379/3'
    sqlite_path: str = str(ROOT / 'data/records.sqlite3')
    chroma_host: str = ''
    chroma_port: int = 8001
    chroma_path: str = str(ROOT / 'data/chroma')
    embedding_base_url: str = 'http://127.0.0.1:1234/v1'
    embedding_model: str = 'text-embedding-bge-m3'
    embedding_api_key: str = 'local'
    knowledge_collection: str = 'knowledge_base__text_embedding_bge_m3'
    case_collection: str = 'resolved_cases_v3'
    rag_recall_k: int = 5
    rag_cache_ttl: int = 300
    retrieval_timeout: float = 30.0
    case_min_similarity: float = 0.35
    dev_api_key: str = ''
    session_signing_key: str = ''
    business_scope: str = 'customer-service-demo'
    skills_dir: str = str(ROOT / 'skills')
    model_call_limit: int = 4
    request_timeout: float = 120.0

    @classmethod
    def from_env(cls):
        load_dotenv(ROOT / '.env')
        mappings = {
            'model': 'ANTHROPIC_MODEL', 'model_base_url': 'ANTHROPIC_BASE_URL',
            'model_api_key': 'ANTHROPIC_API_KEY', 'jev_api_key': 'JEV_API_KEY',
            'jev_model': 'JEV_MODEL', 'jev_endpoint': 'JEV_ENDPOINT',
            'redis_url': 'REDIS_URL', 'embedding_base_url': 'ECHOMIND_EMBEDDING_BASE_URL',
            'embedding_model': 'ECHOMIND_EMBEDDING_MODEL',
            'rag_recall_k': 'ECHOMIND_RAG_RECALL_K', 'dev_api_key': 'DEV_API_KEY',
            'session_signing_key': 'SESSION_SIGNING_KEY', 'business_scope': 'BUSINESS_SCOPE',
            'chroma_host': 'CHROMA_HOST', 'chroma_port': 'CHROMA_PORT',
        }
        instance = cls()
        for field in cls.__dataclass_fields__:
            value = os.getenv(mappings.get(field, field.upper()))
            if value is not None:
                setattr(instance, field, type(getattr(instance, field))(value))
        return instance
