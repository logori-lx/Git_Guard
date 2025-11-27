# File: server/indexer_template.py
# [此文件将被客户端下载并重命名为 git_guard_indexer.py]
import os
import shutil
import chromadb
from typing import List
from git import Repo # 依赖 gitpython 自动定位
from zhipuai import ZhipuAI
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

# --- 关键路径配置 ---
try:
    # 自动向上寻找 .git 所在的文件夹（即项目根目录）
    repo_obj = Repo(".", search_parent_directories=True)
    REPO_PATH = repo_obj.working_tree_dir
except:
    REPO_PATH = "." # Fallback
    
GUARD_DIR = os.path.join(REPO_PATH, ".git_guard")
if not os.path.exists(GUARD_DIR):
    os.makedirs(GUARD_DIR) # 创建 .git_guard 文件夹

# 数据库存放在项目根目录下的 .git_guard 文件夹内（隐藏且随项目存在）
# 这样每个项目都有自己独立的 RAG 库
DB_PATH = os.path.join(REPO_PATH, ".git_guard", "chroma_db")
API_KEY = os.getenv("MEDICAL_RAG") 

# --- 保持不变的配置 ---
LANGUAGE_MAP = {
    ".py": (Language.PYTHON, "repo_python"),
    ".java": (Language.JAVA, "repo_java"),
    ".js": (Language.JS, "repo_js"),
    # ... 其他语言 ...
}

# ... (ZhipuEmbeddingFunction 类保持不变) ...
class ZhipuEmbeddingFunction(chromadb.EmbeddingFunction):
    def __init__(self, api_key):
        self.api_key = api_key
        self.client = ZhipuAI(api_key=api_key)
    def __call__(self, input: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(model="embedding-3", input=input)
        return [data.embedding for data in response.data]

def build_index():
    if not API_KEY:
        # 如果是后台静默运行，打印日志即可
        print("❌ API Key missing. Skipping indexing.")
        return

    print(f"🚀 [Indexer] Scanning: {REPO_PATH}")
    print(f"📂 [Indexer] Database: {DB_PATH}")

    # 注意：在本地更新时，通常我们做增量更新比较复杂。
    # 为了 MVP 稳定性，这里依然采用"全量覆盖"策略。
    # 生产环境可以用增量更新 (Collection.upsert)
    if os.path.exists(DB_PATH):
        try:
            shutil.rmtree(DB_PATH)
        except: pass

    client = chromadb.PersistentClient(path=DB_PATH)
    emb_fn = ZhipuEmbeddingFunction(api_key=API_KEY)

    for suffix, (lang_enum, col_name) in LANGUAGE_MAP.items():
        # ... (这里放入之前修复过的带 fallback 的加载逻辑) ...
        # 为了节省篇幅，核心加载逻辑同上一次回答的修复版
        # 记得把 try-except parser 降级逻辑放进去
        
        # 简写示例：
        loader = GenericLoader.from_filesystem(REPO_PATH, glob=f"**/*{suffix}")
        try:
            docs = loader.load()
        except:
            continue # Skip errors
            
        if not docs: continue
        
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=lang_enum, chunk_size=1000, chunk_overlap=200
        )
        split_docs = splitter.split_documents(docs)
        
        col = client.get_or_create_collection(name=col_name, embedding_function=emb_fn)
        
        # 简单的分批写入
        batch_ids = [f"{col_name}_{i}" for i in range(len(split_docs))]
        batch_texts = [d.page_content for d in split_docs]
        batch_metas = [{"source": d.metadata.get("source", "")} for d in split_docs]
        
        if batch_ids:
            col.add(ids=batch_ids, documents=batch_texts, metadatas=batch_metas)

    print("✅ [Indexer] Local Knowledge Base Updated.")

if __name__ == "__main__":
    build_index()