import os
from pathlib import Path

class Settings:
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent
    CONFIG_DIR = PROJECT_ROOT / "config"
    DOCS_DIR = PROJECT_ROOT / "docs"
    RAW_PDFS_DIR = DOCS_DIR / "raw_pdfs"
    PROCESSED_DIR = DOCS_DIR / "processed"
    DATA_DIR = PROJECT_ROOT / "data"
    LOGS_DIR = PROJECT_ROOT / "logs"
    SRC_DIR = PROJECT_ROOT / "src"
    
    # Configuration files
    AGENTS_CONFIG = CONFIG_DIR / "agents_config.yaml"
    EXTRACTION_CONFIG = CONFIG_DIR / "extraction_config.yaml"
    
    # Vector store paths
    VECTOR_STORES_DIR = DATA_DIR / "vector_stores"
    EMBEDDINGS_CACHE_DIR = DATA_DIR / "embeddings_cache"
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # Ollama settings
    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    
    @classmethod
    def ensure_directories(cls):
        """Create necessary directories"""
        dirs = [
            cls.PROCESSED_DIR / "extracted_text",
            cls.PROCESSED_DIR / "structured_data", 
            cls.PROCESSED_DIR / "embeddings",
            cls.DOCS_DIR / "validation",
            cls.VECTOR_STORES_DIR / "pizza_descriptions",
            cls.VECTOR_STORES_DIR / "allergen_data", 
            cls.VECTOR_STORES_DIR / "recipes",
            cls.VECTOR_STORES_DIR / "nutrition_data",
            cls.EMBEDDINGS_CACHE_DIR,
            cls.LOGS_DIR
        ]
        
        for directory in dirs:
            directory.mkdir(parents=True, exist_ok=True)
            
    @classmethod
    def get_config_path(cls, config_name: str) -> Path:
        """Get path to a configuration file"""
        return cls.CONFIG_DIR / config_name