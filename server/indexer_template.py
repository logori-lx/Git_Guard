# File: server/indexer_template.py
import os
import shutil
import chromadb
from typing import List
from git import Repo
from zhipuai import ZhipuAI
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

# ==========================================
# Config
# ==========================================
try:
    repo_obj = Repo(".", search_parent_directories=True)
    REPO_PATH = repo_obj.working_tree_dir
except:
    REPO_PATH = "." 
    
GUARD_DIR = os.path.join(REPO_PATH, ".git_guard")
if not os.path.exists(GUARD_DIR):
    os.makedirs(GUARD_DIR)

DB_PATH = os.path.join(REPO_PATH, ".git_guard", "chroma_db")
API_KEY = os.getenv("MEDICAL_RAG") 

LANGUAGE_MAP = {
    ".py": (Language.PYTHON, "repo_python"),
    ".java": (Language.JAVA, "repo_java"),
    ".js": (Language.JS, "repo_js"),
    # Add more languages as needed
}

# ==========================================
# Embedding
# ==========================================
class ZhipuEmbeddingFunction(chromadb.EmbeddingFunction):
    def __init__(self, api_key):
        self.api_key = api_key
        self.client = ZhipuAI(api_key=api_key)
    def __call__(self, input: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(model="embedding-3", input=input)
        return [data.embedding for data in response.data]

# ==========================================
# Main Logic: Build Index
# ==========================================
def build_index():
    if not API_KEY:
        print("API Key missing. Skipping indexing.")
        return

    print(f"[Indexer] Scanning: {REPO_PATH}")
    print(f"[Indexer] Database: {DB_PATH}")

    # Full update strategy: Clean and Rebuild
    if os.path.exists(DB_PATH):
        try:
            shutil.rmtree(DB_PATH)
        except: pass

    client = chromadb.PersistentClient(path=DB_PATH)
    emb_fn = ZhipuEmbeddingFunction(api_key=API_KEY)

    for suffix, (lang_enum, col_name) in LANGUAGE_MAP.items():
        parser = None
        try:
            parser = LanguageParser(language=lang_enum, parser_threshold=500)
        except: pass

        # Fallback loading strategy
        if parser:
            loader = GenericLoader.from_filesystem(REPO_PATH, glob=f"**/*{suffix}", parser=parser)
        else:
            loader = GenericLoader.from_filesystem(REPO_PATH, glob=f"**/*{suffix}")

        try:
            docs = loader.load()
        except Exception:
            try:
                # Retry without parser
                loader = GenericLoader.from_filesystem(REPO_PATH, glob=f"**/*{suffix}")
                docs = loader.load()
            except: continue
            
        if not docs: continue
        
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=lang_enum, chunk_size=1000, chunk_overlap=200
        )
        split_docs = splitter.split_documents(docs)
        
        col = client.get_or_create_collection(name=col_name, embedding_function=emb_fn)
        
        batch_ids = [f"{col_name}_{i}" for i in range(len(split_docs))]
        batch_texts = [d.page_content for d in split_docs]
        batch_metas = [{"source": d.metadata.get("source", "")} for d in split_docs]
        
        if batch_ids:
            col.add(ids=batch_ids, documents=batch_texts, metadatas=batch_metas)

    print("[Indexer] Local Knowledge Base Updated.")

if __name__ == "__main__":
    build_index()