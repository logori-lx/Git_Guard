import os
import shutil
import chromadb
from typing import List
from tqdm import tqdm
from zhipuai import ZhipuAI

# LangChain 组件
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

# Configuration
REPO_PATH = "."  
DB_PATH = "./chroma_db"
API_KEY = os.getenv("MEDICAL_RAG") 

# --- 核心配置 ---
LANGUAGE_MAP = {
    ".py":   (Language.PYTHON, "repo_python"),
    ".java": (Language.JAVA,   "repo_java"),
    ".js":   (Language.JS,     "repo_js"),
    ".ts":   (Language.TS,     "repo_js"), 
    ".html": (Language.HTML,   "repo_html"), # HTML 在 Parser 阶段会降级处理
    ".go":   (Language.GO,     "repo_go"),
    ".cpp":  (Language.CPP,    "repo_cpp"),
}

class ZhipuEmbeddingFunction(chromadb.EmbeddingFunction):
    def __init__(self, api_key):
        self.api_key = api_key
        self.client = ZhipuAI(api_key=api_key)

    def __call__(self, input: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(model="embedding-3", input=input)
        return [data.embedding for data in response.data]

def build_index():
    if not API_KEY:
        raise ValueError("❌ Error: API Key environment variable is missing.")

    print(f"🚀 Starting Multi-Language Indexing for: {os.path.abspath(REPO_PATH)}")

    if os.path.exists(DB_PATH):
        try:
            shutil.rmtree(DB_PATH)
            print(f"🧹 Cleaned up existing DB at {DB_PATH}")
        except Exception as e:
            print(f"⚠️ Warning: Could not delete old DB: {e}")

    client = chromadb.PersistentClient(path=DB_PATH)
    emb_fn = ZhipuEmbeddingFunction(api_key=API_KEY)

    # --- 循环处理每种语言 ---
    for suffix, (lang_enum, collection_name) in LANGUAGE_MAP.items():
        print(f"\n📂 Processing {suffix} files for collection: '{collection_name}'...")

        # --- A. 智能加载 (Load) ---
        # 增加容错逻辑：如果 LanguageParser 不支持该语言(如 HTML)，则回退到默认加载器
        parser = None
        try:
            # 尝试初始化高级解析器
            parser = LanguageParser(language=lang_enum, parser_threshold=500)
        except Exception:
            # 捕获 "No parser available" 错误
            print(f"   ℹ️  Note: Advanced parser not available for {suffix}. Using basic text loader.")
            parser = None

        # 根据是否成功初始化 parser 来决定加载方式
        if parser:
            loader = GenericLoader.from_filesystem(
                REPO_PATH,
                glob=f"**/*{suffix}",
                parser=parser
            )
        else:
            # 如果 parser 为空（例如 HTML），不传 parser 参数，GenericLoader 会默认按文本/MIME处理
            loader = GenericLoader.from_filesystem(
                REPO_PATH,
                glob=f"**/*{suffix}"
            )

        documents = loader.load()
        
        if not documents:
            print(f"   ⚠️ No {suffix} files found. Skipping.")
            continue

        print(f"   📄 Loaded {len(documents)} {suffix} documents.")

        # --- B. 切分 (Split) ---
        # 即使 Parser 失败，Splitter 依然支持 HTML 等语言的正则切分，这很好
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=lang_enum, 
            chunk_size=1000, 
            chunk_overlap=200
        )
        split_docs = splitter.split_documents(documents)
        print(f"   🧩 Split into {len(split_docs)} chunks.")

        # --- C. 存储 (Store) ---
        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=emb_fn
        )

        batch_size = 64
        total_docs = len(split_docs)
        
        for start_idx in range(0, total_docs, batch_size):
            end_idx = min(start_idx + batch_size, total_docs)
            batch = split_docs[start_idx:end_idx]
            
            batch_ids = []
            batch_documents = []
            batch_metadatas = []
            
            for i, doc in enumerate(batch):
                batch_documents.append(doc.page_content)
                
                meta = doc.metadata.copy()
                meta["language"] = str(lang_enum) 
                for k, v in meta.items():
                    if v is None: meta[k] = ""
                batch_metadatas.append(meta)
                
                safe_name = os.path.basename(meta.get('source', 'unknown')).replace('.', '_')
                unique_id = f"{collection_name}_{safe_name}_{start_idx + i}"
                batch_ids.append(unique_id)
            
            if batch_ids:
                collection.add(ids=batch_ids, documents=batch_documents, metadatas=batch_metadatas)

    print(f"\n✅ All Languages Indexed Successfully into {DB_PATH}")

if __name__ == "__main__":
    build_index()