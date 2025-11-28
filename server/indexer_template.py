# File: server/indexer_template.py
import os
import sys
import shutil
import requests
import chromadb
from typing import List, Dict, Any
from git import Repo
from zai import ZhipuAiClient

# LangChain 组件
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

# ==========================================
# 1. 基础配置与路径 Fix
# ==========================================

# Windows GBK 编码修复
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError: pass

try:
    repo_obj = Repo(".", search_parent_directories=True)
    REPO_PATH = repo_obj.working_tree_dir
except:
    REPO_PATH = "." 

GUARD_DIR = os.path.join(REPO_PATH, ".git_guard")
if not os.path.exists(GUARD_DIR):
    try: os.makedirs(GUARD_DIR)
    except: pass

DB_PATH = os.path.join(GUARD_DIR, "chroma_db")
API_KEY = os.getenv("ZHIPU_API_KEY") 

EXT_TO_COLLECTION = {
    ".py": "repo_python", ".java": "repo_java", ".js": "repo_js",
    ".ts": "repo_js", ".html": "repo_html", ".go": "repo_go", ".cpp": "repo_cpp"
}

# 必须保留 LANGUAGE_MAP 供 build_index 使用
LANGUAGE_MAP = {
    ".py": (Language.PYTHON, "repo_python"),
    ".java": (Language.JAVA, "repo_java"),
    ".js": (Language.JS, "repo_js"),
    ".ts": (Language.TS, "repo_js"),
    ".html": (Language.HTML, "repo_html"),
    ".go": (Language.GO, "repo_go"),
    ".cpp": (Language.CPP, "repo_cpp")
}

# ==========================================
# 2. 核心类定义
# ==========================================

class Reranker:
    def __init__(self):
        self.url = "https://open.bigmodel.cn/api/paas/v4/rerank"
        self.model = "rerank-3"
        self.api_key = API_KEY

    def rerank(self, query: str, documents: List[Dict], top_k: int = 3) -> List[Dict]:
        if not self.api_key or not documents:
            return documents[:top_k]

        doc_texts = [doc.get("answer", "")[:2000] for doc in documents]

        payload = {
            "model": self.model,
            "query": query[:1000], 
            "documents": doc_texts,
            "top_n": top_k
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(self.url, json=payload, headers=headers, timeout=5)
            if response.status_code == 200:
                results = response.json().get('results', [])
                reranked_docs = []
                for item in results:
                    original_idx = item['index']
                    doc = documents[original_idx]
                    doc['score'] = item['relevance_score']
                    reranked_docs.append(doc)
                return reranked_docs
            else:
                return documents[:top_k]
        except Exception:
            return documents[:top_k]

class ZhipuEmbeddingFunction(chromadb.EmbeddingFunction):
    def __init__(self):
        self.api_key = API_KEY
        self.client = ZhipuAiClient(api_key=self.api_key)
    
    def __call__(self, input: List[str]) -> List[List[float]]:
        if not self.api_key: return [[]] * len(input)
        try:
            # 这里的 batch size 由 ChromaDB 外部调用者控制，或者我们在 __call__ 里自己分批
            # 但最好的做法是在 collection.add 层面控制
            response = self.client.embeddings.create(model="embedding-3", input=input)
            return [data.embedding for data in response.data]
        except Exception as e:
            print(f"[Error] Embedding failed: {e}")
            # 返回空向量防止程序崩溃 (维度需匹配，这里假设是 1024 或 2048，暂时返回空列表会报错，只能抛出)
            raise e

class Retrieval:
    def __init__(self):
        if not os.path.exists(DB_PATH):
            self.client = None
            return

        self.client = chromadb.PersistentClient(path=DB_PATH)
        self.embedding_function = ZhipuEmbeddingFunction()
        self.vector_distance_max = 2.0

    def vector_retrieve(self, query: str, collection_name: str, top_k: int = 5) -> List[Dict]:
        if not self.client: return []
        try:
            collection = self.client.get_collection(
                name=collection_name, 
                embedding_function=self.embedding_function
            )
            results = collection.query(query_texts=[query], n_results=top_k)
            
            hits = []
            if results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    dist = results['distances'][0][i]
                    score = 1 - min(dist / self.vector_distance_max, 1.0)
                    hits.append({
                        "id": results['ids'][0][i],
                        "answer": results['documents'][0][i],
                        "metadata": results['metadatas'][0][i],
                        "score": score,
                        "source": "vector"
                    })
            return hits
        except Exception:
            return []

    def hybrid_retrieve(self, query: str, collection_name: str, top_k: int = 5) -> List[Dict]:
        vector_hits = self.vector_retrieve(query, collection_name, top_k=top_k * 2)
        keywords = set(query.split())
        for hit in vector_hits:
            code_content = hit["answer"]
            match_count = sum(1 for kw in keywords if kw in code_content)
            keyword_score = min(match_count * 0.1, 1.0)
            hit["score"] = (hit["score"] * 0.7) + (keyword_score * 0.3)
            hit["source"] = "hybrid"
        sorted_hits = sorted(vector_hits, key=lambda x: x["score"], reverse=True)[:top_k]
        return sorted_hits

    def retrieve_code(self, query_diff: str, file_ext: str, top_k: int = 5) -> List[Dict]:
        if file_ext not in EXT_TO_COLLECTION: return []
        col_name = EXT_TO_COLLECTION[file_ext]
        return self.hybrid_retrieve(query_diff, col_name, top_k)

# ==========================================
# 3. 建库逻辑 (Build Index)
# ==========================================

def build_index():
    if not API_KEY:
        print("API Key missing. Skipping indexing.")
        return

    print(f"[Indexer] Scanning: {REPO_PATH}")
    print(f"[Indexer] Database: {DB_PATH}")

    if os.path.exists(DB_PATH):
        try: shutil.rmtree(DB_PATH)
        except: pass

    client = chromadb.PersistentClient(path=DB_PATH)
    emb_fn = ZhipuEmbeddingFunction()

    for suffix, (lang_enum, col_name) in LANGUAGE_MAP.items():
        # ... (加载逻辑不变) ...
        parser = None
        try: parser = LanguageParser(language=lang_enum, parser_threshold=500)
        except: pass

        if parser:
            loader = GenericLoader.from_filesystem(REPO_PATH, glob=f"**/*{suffix}", parser=parser)
        else:
            loader = GenericLoader.from_filesystem(REPO_PATH, glob=f"**/*{suffix}")

        try:
            docs = loader.load()
        except Exception:
            try:
                loader = GenericLoader.from_filesystem(REPO_PATH, glob=f"**/*{suffix}")
                docs = loader.load()
            except: continue
            
        if not docs: continue
        
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=lang_enum, chunk_size=1000, chunk_overlap=200
        )
        split_docs = splitter.split_documents(docs)
        
        col = client.get_or_create_collection(name=col_name, embedding_function=emb_fn)
    
        BATCH_SIZE = 50 
        
        total_docs = len(split_docs)
        print(f"   -> Found {total_docs} chunks for {suffix}. Processing in batches...")

        for i in range(0, total_docs, BATCH_SIZE):
            batch = split_docs[i : i + BATCH_SIZE]
            
            batch_ids = [f"{col_name}_{i+j}" for j in range(len(batch))]
            batch_texts = [d.page_content for d in batch]
            batch_metas = []
            
            for d in batch:
                meta = d.metadata.copy()
                for k, v in meta.items():
                    if v is None: meta[k] = ""
                batch_metas.append(meta)
            
            try:
                col.add(ids=batch_ids, documents=batch_texts, metadatas=batch_metas)
            except Exception as e:
                print(f"      [Error] Failed to add batch {i}: {e}")

    print("[Indexer] Local Knowledge Base Updated.")

if __name__ == "__main__":
    build_index()