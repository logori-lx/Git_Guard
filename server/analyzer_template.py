# File: server/analyzer_template.py
import os
import sys
import re
import requests
import chromadb
from typing import List, Dict, Any
from git import Repo
from zhipuai import ZhipuAI

# ==========================================
# 1. 基础环境与配置
# ==========================================

# [FIX] Windows GBK 编码修复
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

SERVER_BASE_URL = "http://localhost:8000"
CONFIG_URL = f"{SERVER_BASE_URL}/api/v1/config"
TRACK_URL = f"{SERVER_BASE_URL}/api/v1/track"

# 优先读取环境变量，如果没有则尝试从系统读取
API_KEY = os.getenv("MEDICAL_RAG")

# 自动定位项目根目录
try:
    repo_obj = Repo(".", search_parent_directories=True)
    REPO_PATH = repo_obj.working_tree_dir
except:
    REPO_PATH = "."

# 数据库路径
GUARD_DIR = os.path.join(REPO_PATH, ".git_guard")
DB_PATH = os.path.join(GUARD_DIR, "chroma_db")

# 文件后缀 -> 集合名称 映射
EXT_TO_COLLECTION = {
    ".py": "repo_python", ".java": "repo_java", ".js": "repo_js",
    ".ts": "repo_js", ".html": "repo_html", ".go": "repo_go", 
    ".cpp": "repo_cpp", ".c": "repo_cpp"
}

# ==========================================
# 2. 核心类定义 (从原 rerank.py 和 retrieval.py 合并而来)
# ==========================================

class Reranker:
    """调用智谱 AI 进行语义重排序"""
    def __init__(self):
        self.url = "https://open.bigmodel.cn/api/paas/v4/rerank"
        self.model = "rerank-3"
        self.api_key = API_KEY

    def rerank(self, query: str, documents: List[Dict], top_k: int = 3) -> List[Dict]:
        """
        :param documents: List of dicts, must contain 'answer' key
        """
        if not self.api_key or not documents:
            return documents[:top_k]

        # 提取纯文本用于 API 调用 (截断防止超长)
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
                    doc['score'] = item['relevance_score'] # 更新分数为 Rerank 分数
                    reranked_docs.append(doc)
                return reranked_docs
            else:
                return documents[:top_k]
        except Exception:
            return documents[:top_k]

class ZhipuEmbeddingFunction(chromadb.EmbeddingFunction):
    """智谱 Embedding 适配器"""
    def __init__(self):
        self.api_key = API_KEY
        self.client = ZhipuAI(api_key=self.api_key)
    
    def __call__(self, input: List[str]) -> List[List[float]]:
        if not self.api_key: return [[]] * len(input)
        try:
            response = self.client.embeddings.create(model="embedding-3", input=input)
            return [data.embedding for data in response.data]
        except:
            return [[]] * len(input)

class Retrieval:
    """混合检索管理器 (Vector + Keyword)"""
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
        # 1. 向量召回 (扩大范围)
        vector_hits = self.vector_retrieve(query, collection_name, top_k=top_k * 2)
        
        # 2. 关键词加权 (简单的字面匹配)
        keywords = set(query.split())
        for hit in vector_hits:
            code_content = hit["answer"]
            match_count = sum(1 for kw in keywords if kw in code_content)
            # 混合打分: Vector(0.7) + Keyword(0.3)
            keyword_score = min(match_count * 0.1, 1.0)
            hit["score"] = (hit["score"] * 0.7) + (keyword_score * 0.3)
            hit["source"] = "hybrid"

        # 3. 排序截断
        sorted_hits = sorted(vector_hits, key=lambda x: x["score"], reverse=True)[:top_k]
        return sorted_hits

    def retrieve_code(self, query_diff: str, file_ext: str, top_k: int = 5) -> List[Dict]:
        if file_ext not in EXT_TO_COLLECTION: return []
        col_name = EXT_TO_COLLECTION[file_ext]
        return self.hybrid_retrieve(query_diff, col_name, top_k)

# ==========================================
# 3. 辅助功能函数
# ==========================================

def get_console_input(prompt_text):
    print(prompt_text, end='', flush=True)
    try:
        if sys.platform == 'win32':
            with open('CON', 'r', encoding='utf-8') as f: return f.readline().strip()
        else:
            with open('/dev/tty', 'r', encoding='utf-8') as f: return f.readline().strip()
    except: return input().strip()

def fetch_dynamic_rules():
    try:
        resp = requests.get(CONFIG_URL, timeout=1.5)
        if resp.status_code == 200: return resp.json()
    except: pass
    return {"template_format": "Standard", "custom_rules": "None"}

def report_to_cloud(msg, risk, summary):
    try:
        user = os.getenv("USERNAME") or "Unknown"
        payload = {
            "developer_id": user,
            "repo_name": os.path.basename(os.path.abspath(REPO_PATH)),
            "commit_msg": msg,
            "risk_level": risk,
            "ai_summary": summary
        }
        requests.post(TRACK_URL, json=payload, timeout=2)
    except: pass

# ==========================================
# 4. 业务逻辑：Hybrid Retrieve + Rerank + LLM
# ==========================================

def process_changes_with_rag():
    if not API_KEY: return {}, ""

    try:
        repo = Repo(REPO_PATH)
        try:
            diff_index = repo.head.commit.diff()
        except ValueError:
            EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"
            diff_index = repo.tree(EMPTY_TREE).diff(repo.index)
    except: return {}, ""

    changes = {}
    context_str = ""
    
    # 实例化刚才定义的类
    retriever = Retrieval()
    reranker = Reranker()

    for diff in diff_index:
        if diff.change_type == 'D': continue
        fpath = diff.b_path if diff.b_path else diff.a_path
        if not fpath: continue
        
        try:
            text = repo.git.diff("--cached", fpath)
            if not text.strip(): text = "(New File)"
            _, ext = os.path.splitext(fpath)
            
            changes[fpath] = text

            # --- RAG 流程 ---
            # 1. 混合检索
            candidates = retriever.retrieve_code(query_diff=text, file_ext=ext, top_k=10)
            
            if candidates:
                # 2. 重排序
                final_docs = reranker.rerank(query=text, documents=candidates, top_k=3)
                
                for doc in final_docs:
                    score = doc.get('score', 0)
                    content = doc.get('answer', '')[:500]
                    context_str += f"\n[Ref Score: {score:.2f}]:\n{content}\n"

        except Exception: pass

    return changes, context_str

def run(msg_file_path):
    try:
        with open(msg_file_path, 'r', encoding='utf-8') as f:
            original_msg = f.read().strip()
    except: return
    
    if not original_msg: return

    print(f"🔄 [Git-Guard] Analyzing (Hybrid RAG + Rerank)...")
    
    changes, context = process_changes_with_rag()
    
    if not changes: return

    config = fetch_dynamic_rules()
    fmt = config.get("template_format", "Standard")
    rules = config.get("custom_rules", "")

    # LLM Prompt
    prompt = f"""
    Role: Code Reviewer & Commit Message Generator.
    
    User Draft: "{original_msg}"
    
    Code Changes: 
    {str(list(changes.values()))[:3000]}
    
    Knowledge Context:
    {context[:1500]}
    
    >>> RULES <<<
    Template: "{fmt}"
    Instructions: "{rules}"
    >>> END RULES <<<
    
    STRICT OUTPUT FORMAT:
    RISK: <High/Medium/Low>
    SUMMARY: <Summary>
    OPTIONS: <Msg1>|||<Msg2>|||<Msg3>
    
    Example:
    OPTIONS: [Backend] fix login|||fix: auth bug|||refactor: login
    """

    try:
        client = ZhipuAI(api_key=API_KEY)
        res = client.chat.completions.create(
            model="glm-4-flash", 
            messages=[{"role": "user", "content": prompt}]
        )
        content = res.choices[0].message.content
        
        # 解析逻辑
        risk = "Medium"
        summary = "Update"
        options = []
        
        for line in content.split('\n'):
            if line.startswith("RISK:"): risk = line.replace("RISK:", "").strip()
            if line.startswith("SUMMARY:"): summary = line.replace("SUMMARY:", "").strip()
            if "OPTIONS:" in line:
                raw = line.split("OPTIONS:")[1].strip()
                options = [p.strip() for p in raw.split('|||') if p.strip()]

        final_options = []
        for opt in options:
            opt = re.sub(r'^[\d\-\.\s]+', '', opt).replace("OPTIONS:", "").strip()
            if len(opt) > 3: final_options.append(opt)
        
        while len(final_options) < 3: final_options.append(f"refactor: {original_msg}")
        options = final_options[:3]

    except Exception as e:
        print(f"AI Failed: {e}")
        return

    # 交互界面
    print("\n" + "="*60)
    print(f"🤖 AI SUGGESTIONS (Risk: {risk})")
    print("="*60)
    print(f"[0] [Keep Original]: {original_msg}")
    print(f"[1] {options[0]}")
    print(f"[2] {options[1]}")
    print(f"[3] {options[2]}")
    print("="*60)

    sel = get_console_input("\n👉 Select (0-3): ")
    
    final_msg = original_msg
    if sel == '1': final_msg = options[0]
    elif sel == '2': final_msg = options[1]
    elif sel == '3': final_msg = options[2]

    if final_msg != original_msg:
        with open(msg_file_path, 'w', encoding='utf-8') as f:
            f.write(final_msg)
        print("✅ Updated.")

    report_to_cloud(final_msg, risk, summary)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run(sys.argv[1])