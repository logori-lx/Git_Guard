# File: server/analyzer_template.py
import os
import sys
import re
import requests
import chromadb
from typing import List, Dict, Any
from git import Repo
from zhipuai import ZhipuAI
import getpass

# ==========================================
# Config & Environment
# ==========================================

# Fix Windows console encoding issues
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

SERVER_BASE_URL = "http://localhost:8000"
CONFIG_URL = f"{SERVER_BASE_URL}/api/v1/config"
TRACK_URL = f"{SERVER_BASE_URL}/api/v1/track"

API_KEY = os.getenv("MEDICAL_RAG")

try:
    repo_obj = Repo(".", search_parent_directories=True)
    REPO_PATH = repo_obj.working_tree_dir
except:
    REPO_PATH = "."

GUARD_DIR = os.path.join(REPO_PATH, ".git_guard")
DB_PATH = os.path.join(GUARD_DIR, "chroma_db")

EXT_TO_COLLECTION = {
    ".py": "repo_python", ".java": "repo_java", ".js": "repo_js",
    ".ts": "repo_js", ".html": "repo_html", ".go": "repo_go", 
    ".cpp": "repo_cpp", ".c": "repo_cpp"
}

# ==========================================
# Core Classes: Rerank & Retrieval
# ==========================================

class Reranker:
    """Zhipu AI Semantic Reranker"""
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
        self.client = ZhipuAI(api_key=self.api_key)
    
    def __call__(self, input: List[str]) -> List[List[float]]:
        if not self.api_key: return [[]] * len(input)
        try:
            response = self.client.embeddings.create(model="embedding-3", input=input)
            return [data.embedding for data in response.data]
        except:
            return [[]] * len(input)

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
# Helpers
# ==========================================

def get_abort_flag_path():
    return os.path.join(GUARD_DIR, "abort_commit.flag")

def clean_markdown(text):
    """Strip Markdown formatting for terminal output"""
    filtered_text = ""
    for char in text:
        if char in ('*', '`', '#'): continue
        filtered_text += char
    return filtered_text.strip()

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
        try:
            user = getpass.getuser()
        except Exception:
            user = os.getenv("USERNAME") or os.getenv("USER") or "Unknown"

        payload = {
            "developer_id": user,
            "repo_name": os.path.basename(os.path.abspath(REPO_PATH)),
            "commit_msg": msg,
            "risk_level": risk,
            "ai_summary": summary
        }
        requests.post(TRACK_URL, json=payload, timeout=2)
    except Exception:
        pass

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
            candidates = retriever.retrieve_code(query_diff=text, file_ext=ext, top_k=10)
            if candidates:
                final_docs = reranker.rerank(query=text, documents=candidates, top_k=3)
                for doc in final_docs:
                    score = doc.get('score', 0)
                    content = doc.get('answer', '')[:500]
                    context_str += f"\n[Ref Score: {score:.2f}]:\n{content}\n"
        except Exception: pass

    return changes, context_str

# ==========================================
# Mode 1: Pre-commit Report (Report Only)
# ==========================================
def run_report_mode():
    flag_path = get_abort_flag_path()
    if os.path.exists(flag_path):
        os.remove(flag_path)

    print(f"[Git-Guard] Repo: {os.path.abspath(REPO_PATH)}")
    
    changes, context = process_changes_with_rag()
    
    if not changes:
        print("[Info] No staged changes to analyze.")
        return

    print("[Git-Guard] Analyzing Impact & Risk (RAG Enhanced)...")
    
    prompt = f"""
    Role: Senior Technical Lead conducting a Pre-commit Risk Assessment.
    Code Changes: {str(list(changes.values()))[:3000]}
    Context: {context[:1500]}
    Task: Generate a concise impact report.
    STRICT OUTPUT FORMAT (Plain Text):
    RISK LEVEL: <High/Medium/Low>
    ------------------------------------------------------------
    IMPACT ANALYSIS:
    - <Point 1>
    - <Point 2>
    ------------------------------------------------------------
    Constraint: Do NOT use Markdown formatting. Keep it short.
    """
    
    try:
        client = ZhipuAI(api_key=API_KEY)
        res = client.chat.completions.create(
            model="glm-4-flash", 
            messages=[{"role": "user", "content": prompt}]
        )
        report = clean_markdown(res.choices[0].message.content)
        
        print("\n" + "="*60)
        print(" GIT-GUARD IMPACT REPORT")
        print("-" * 60)
        print(report)
        print("="*60 + "\n")
        
        if "RISK LEVEL: High" in report:
            print("[Warning] High Risk Detected! Please review carefully.")
            
    except Exception as e:
        print(f"[Error] Analysis failed: {e}")
        return

    print("\n[?] Do you want to proceed with this commit? [Y/n]: ", end="", flush=True)
    choice = get_console_input("").lower()

    if choice == 'n':
        print("\n[Abort] Commit aborted by user.")
        with open(flag_path, 'w') as f:
            f.write("aborted")
        sys.exit(1)
    else:
        print("[Info] Proceeding to commit message generation...")

# ==========================================
# Mode 2: Commit-Msg (Interactive Suggestion)
# ==========================================
def run_suggestion_mode(msg_file_path):
    if os.path.exists(get_abort_flag_path()):
        sys.exit(1)
    
    try:
        with open(msg_file_path, 'r', encoding='utf-8') as f:
            original_msg = f.read().strip()
    except: return
    
    if not original_msg: return

    changes, context = process_changes_with_rag()
    if not changes: return

    config = fetch_dynamic_rules()
    fmt = config.get("template_format", "Standard")
    rules = config.get("custom_rules", "")

    prompt = f"""
    Role: Commit Message Generator.
    User Draft: "{original_msg}"
    Code Changes: {str(list(changes.values()))[:3000]} 
    Context: {context[:1500]}
    
    >>> RULES <<<
    Template: "{fmt}"
    Instructions: "{rules}"
    >>> END RULES <<<
    
    STRICT FORMAT:
    RISK: <Level>
    SUMMARY: <Summary>
    OPTIONS: <Msg1>|||<Msg2>|||<Msg3>
    
    Constraints: Plain text only. No Markdown. No numbered lists.
    """

    try:
        client = ZhipuAI(api_key=API_KEY)
        res = client.chat.completions.create(
            model="glm-4-flash", 
            messages=[{"role": "user", "content": prompt}]
        )
        content = res.choices[0].message.content
        
        risk = "Medium"
        summary = "Update"
        options = []
        
        for line in content.split('\n'):
            line = clean_markdown(line)
            if line.startswith("RISK:"): risk = line.replace("RISK:", "").strip()
            if line.startswith("SUMMARY:"): summary = line.replace("SUMMARY:", "").strip()
            if "OPTIONS:" in line:
                raw = line.split("OPTIONS:")[1].strip()
                options = [p.strip() for p in raw.split('|||') if p.strip()]

        final_options = []
        for opt in options:
            opt = clean_markdown(opt)
            opt = re.sub(r'^[\d\-\.\s]+', '', opt).replace("OPTIONS:", "").strip()
            if len(opt) > 3: final_options.append(opt)
        
        while len(final_options) < 3: final_options.append(f"refactor: {original_msg}")
        options = final_options[:3]

    except Exception: return

    print("\n" + "="*60)
    print(f" COMMIT SUGGESTIONS (Risk: {risk})")
    print("-" * 60)
    print(f"[0] [Keep Original]: {original_msg}")
    print(f"[1] {options[0]}")
    print(f"[2] {options[1]}")
    print(f"[3] {options[2]}")
    print("="*60)

    sel = get_console_input("\n[?] Select (0-3): ")
    
    final_msg = original_msg
    if sel == '1': final_msg = options[0]
    elif sel == '2': final_msg = options[1]
    elif sel == '3': final_msg = options[2]

    if final_msg != original_msg:
        with open(msg_file_path, 'w', encoding='utf-8') as f:
            f.write(final_msg)
        print("[Success] Updated.")

    report_to_cloud(final_msg, risk, summary)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_suggestion_mode(sys.argv[1])
    else:
        run_report_mode()