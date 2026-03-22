import os


class Config:
    # LLM Settings
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llava:7b")
    USE_VISION = True
    LLM_THRESHOLD_DISTANCE = 100.0
    LLM_TIMEOUT = 30  # seconds

    # FAISS Settings
    FAISS_INDEX_PATH = "vectordb/index.faiss"
    IMAGE_IDS_PATH = "vectordb/image_ids.npy"

    # Database
    DATABASE_URL = "postgresql://steel_defects_db:steel_defects_db@localhost/steel_defects_db"

    # Image Processing
    IMAGE_SIZE = (224, 224)
    EMBEDDING_DIM = 2048
