# File: server/analyzer_template.py
import os
import sys
import re  # [新增] 用于清洗 AI 输出
import requests 
import chromadb
from typing import List
from git import Repo
from zhipuai import ZhipuAI

# [FIX] Windows GBK 编码修复
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# --- 配置 ---
CLOUD_SERVER_URL = "http://localhost:8000/api/v1/track"

try:
    repo_obj = Repo(".", search_parent_directories=True)
    REPO_PATH = repo_obj.working_tree_dir
except:
    REPO_PATH = "."

# [关键修复] 确保父目录存在
GUARD_DIR = os.path.join(REPO_PATH, ".git_guard")
if not os.path.exists(GUARD_DIR):
    os.makedirs(GUARD_DIR)

DB_PATH = os.path.join(GUARD_DIR, "chroma_db")
API_KEY = os.getenv("MEDICAL_RAG") 

EXT_TO_COLLECTION = {
    ".py": "repo_python", ".java": "repo_java", ".js": "repo_js",
    ".ts": "repo_js", ".html": "repo_html", ".go": "repo_go", ".cpp": "repo_cpp"
}

# --- 辅助类与函数 ---

class ZhipuEmbeddingFunction(chromadb.EmbeddingFunction):
    def __init__(self, api_key):
        self.api_key = api_key
        self.client = ZhipuAI(api_key=api_key)
    def __call__(self, input: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(model="embedding-3", input=input)
        return [data.embedding for data in response.data]

def get_console_input(prompt_text):
    print(prompt_text, end='', flush=True)
    try:
        if sys.platform == 'win32':
            with open('CON', 'r') as f: return f.readline().strip()
        else:
            with open('/dev/tty', 'r') as f: return f.readline().strip()
    except:
        return input().strip()

def get_diff_and_context():
    if not API_KEY: return None, None
    try:
        repo = Repo(REPO_PATH)
        try:
            diff_index = repo.head.commit.diff()
        except ValueError:
            EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"
            diff_index = repo.tree(EMPTY_TREE).diff(repo.index)
    except:
        return None, None

    changes = {}
    for diff in diff_index:
        if diff.change_type == 'D': continue
        fpath = diff.b_path if diff.b_path else diff.a_path
        if not fpath: continue
        
        _, ext = os.path.splitext(fpath)
        if ext in EXT_TO_COLLECTION:
            col = EXT_TO_COLLECTION[ext]
            try:
                text = repo.git.diff("--cached", fpath)
                if not text.strip(): text = "(New File/Content Unavailable)"
                if col not in changes: changes[col] = ""
                changes[col] += f"\nFile: {fpath}\n{text}\n"
            except: pass

    context = ""
    if os.path.exists(DB_PATH) and changes:
        try:
            client = chromadb.PersistentClient(path=DB_PATH)
            emb = ZhipuEmbeddingFunction(api_key=API_KEY)
            for col_name, content in changes.items():
                try:
                    col = client.get_collection(name=col_name, embedding_function=emb)
                    res = col.query(query_texts=[content], n_results=2)
                    if res['documents']:
                        for doc in res['documents'][0]:
                            context += f"\nContext ({col_name}):\n{doc[:300]}...\n"
                except: pass
        except Exception:
            pass
            
    return changes, context

def report_to_cloud(msg, risk, summary):
    try:
        user = os.getenv("USERNAME") or os.getenv("USER") or "Unknown Developer"
        payload = {
            "developer_id": user,
            "repo_name": os.path.basename(os.path.abspath(REPO_PATH)),
            "commit_msg": msg,
            "risk_level": risk,
            "ai_summary": summary
        }
        requests.post(CLOUD_SERVER_URL, json=payload, timeout=2)
    except:
        pass

# --- 主逻辑 ---
def run(msg_file_path):
    with open(msg_file_path, 'r', encoding='utf-8') as f:
        original_msg = f.read().strip()
    
    if not original_msg: return

    print(f"🔄 [Git-Guard] Analyzing changes...")
    changes, context = get_diff_and_context()
    
    if not changes: 
        return

    # [优化] 强化 Prompt，强制单行输出，禁止编号
    prompt = f"""
    User Draft: "{original_msg}"
    Code Changes: {list(changes.values())}
    Context: {context[:1000]}
    
    Task: Generate 3 commit messages.
    
    STRICT FORMAT:
    RISK: <Level>
    SUMMARY: <Summary>
    OPTIONS: <Msg1>|||<Msg2>|||<Msg3>
    
    RULES:
    1. Do NOT use numbered lists (1., 2.).
    2. Do NOT use newlines between options.
    3. Use '|||' as the ONLY separator.
    4. Provide strictly 3 options.
    """
    
    try:
        client = ZhipuAI(api_key=API_KEY)
        res = client.chat.completions.create(
            model="glm-4-flash", messages=[{"role": "user", "content": prompt}]
        )
        content = res.choices[0].message.content
        
        # 解析 AI 返回
        risk_level = "Medium"
        summary = "Code update"
        options = []
        
        for line in content.split('\n'):
            clean_line = line.strip()
            if clean_line.startswith("RISK:"): 
                risk_level = clean_line.replace("RISK:", "").strip()
            if clean_line.startswith("SUMMARY:"): 
                summary = clean_line.replace("SUMMARY:", "").strip()
            
            # [优化] 解析逻辑：处理 AI 可能的换行或格式错误
            if "OPTIONS:" in clean_line:
                raw_opts = clean_line.split("OPTIONS:")[1].strip()
                # 如果 AI 没有换行，直接分割
                if "|||" in raw_opts:
                    parts = raw_opts.split('|||')
                    options = [p.strip() for p in parts if p.strip()]
        
        # [补救] 如果上面没解析到，尝试全文搜索 |||
        if not options and "|||" in content:
             # 尝试提取最后一部分
             parts = content.split('|||')
             # 清洗数据：去掉 AI 可能加上的 "OPTIONS:" 前缀
             options = [p.replace("OPTIONS:", "").strip() for p in parts if len(p.strip()) > 5]

        # [兜底] 如果还是空的，或者 AI 真的输出了编号，进行正则清洗
        final_options = []
        for opt in options:
            # 去掉开头的 "1. ", "2. ", "- " 等
            clean_opt = re.sub(r'^[\d\-\.\s]+', '', opt)
            if clean_opt:
                final_options.append(clean_opt)
        
        # 补齐选项
        while len(final_options) < 3: 
            final_options.append("refactor: update code structure")
            
        options = final_options[:3]

    except Exception as e:
        print(f"⚠️ AI Analysis failed: {e}")
        return

    # 3. 交互式选择 (数字风格)
    print("\n" + "="*60)
    print(f"🤖 AI SUGGESTIONS (Risk: {risk_level})")
    print("="*60)
    print(f"[0] [Keep Original]: {original_msg}")
    print(f"[1] {options[0]}")
    print(f"[2] {options[1]}")
    print(f"[3] {options[2]}")
    print("="*60)

    selection = get_console_input("\n👉 Select (0-3) [Enter for 0]: ")

    final_msg = original_msg
    if selection == '1': final_msg = options[0]
    elif selection == '2': final_msg = options[1]
    elif selection == '3': final_msg = options[2]
    
    if final_msg != original_msg:
        with open(msg_file_path, 'w', encoding='utf-8') as f:
            f.write(final_msg)
        print(f"✅ Message updated.")

    report_to_cloud(final_msg, risk_level, summary)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run(sys.argv[1])