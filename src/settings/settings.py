import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Define multiple paths for the combined corpus
DATA_DIR = os.path.join(BASE_DIR, "../data/retrieval")
DATA_PATHS = {
    "train": os.path.join(DATA_DIR, "train_data.json"),
    "dev": os.path.join(DATA_DIR, "dev_data.json"),
    "test": os.path.join(DATA_DIR, "test_data.json")
}
EXTRACTION_DATA_DIR = os.path.join(BASE_DIR, "../data/extraction")
EXTRACTION_DATA_PATHS = {
    "train": os.path.join(EXTRACTION_DATA_DIR, "train_synthesis.json"),
    "dev": os.path.join(EXTRACTION_DATA_DIR, "dev_synthesis.json"),
    "test": os.path.join(EXTRACTION_DATA_DIR, "test_synthesis.json")
}
EXTRACTOR_MODEL_PATH = os.path.join(BASE_DIR, "weights/BERTSum/bertext_cnndm_transformer.pt")
STORAGE_DIR = os.path.join(BASE_DIR, "vector_db_storage")
EMBEDDING_MODEL = "AITeamVN/Vietnamese_Embedding"
TRUNCATION_DIM = 1024