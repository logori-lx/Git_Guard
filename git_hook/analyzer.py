import os
import sys
import chromadb
from typing import List, Dict, Optional
from git import Repo
from zhipuai import ZhipuAI

# --- Configuration ---
try:
    repo_root = Repo(".", search_parent_directories=True).working_tree_dir
    REPO_PATH = repo_root
except:
    REPO_PATH = "."

DB_PATH = os.path.join(REPO_PATH, "git_hook", "chroma_db")
API_KEY = os.getenv("MEDICAL_RAG")

EXT_TO_COLLECTION = {
    ".py": "repo_python", ".java": "repo_java", ".js": "repo_js",
    ".ts": "repo_js", ".html": "repo_html", ".go": "repo_go", ".cpp": "repo_cpp"
}

# --- Helpers ---

class ZhipuEmbeddingFunction(chromadb.EmbeddingFunction):
    def __init__(self, api_key):
        self.api_key = api_key
        self.client = ZhipuAI(api_key=api_key)
    def __call__(self, input: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(model="embedding-3", input=input)
        return [data.embedding for data in response.data]

def get_console_input(prompt_text):
    """
    强制从终端读取输入，绕过 Git Hook 的 stdin 限制
    """
    print(prompt_text, end='', flush=True)
    try:
        # Windows 使用 CON，Linux/Mac 使用 /dev/tty
        if sys.platform == 'win32':
            with open('CON', 'r') as f:
                return f.readline().strip()
        else:
            with open('/dev/tty', 'r') as f:
                return f.readline().strip()
    except Exception:
        # 如果失败，回退到标准 input (可能在某些终端会卡住)
        return input().strip()

def get_diff_and_context():
    """
    获取 diff 内容和 RAG 上下文，供两个模式复用
    """
    if not API_KEY: return None, None
    
    try:
        repo = Repo(REPO_PATH)
        # 针对 Initial Commit 的处理
        try:
            diff_index = repo.head.commit.diff()
        except ValueError:
            EMPTY_TREE_SHA = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"
            diff_index = repo.tree(EMPTY_TREE_SHA).diff(repo.index)
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
                if not text.strip(): text = "(New File)"
                if col not in changes: changes[col] = ""
                changes[col] += f"\nFile: {fpath}\n{text}\n"
            except: pass

    if not changes: return None, None

    # RAG Search
    client = chromadb.PersistentClient(path=DB_PATH)
    emb = ZhipuEmbeddingFunction(api_key=API_KEY)
    context = ""
    
    for col_name, content in changes.items():
        try:
            col = client.get_collection(name=col_name, embedding_function=emb)
            res = col.query(query_texts=[content], n_results=2)
            if res['documents']:
                for doc in res['documents'][0]:
                    context += f"\nContext ({col_name}):\n{doc[:500]}...\n"
        except: pass
        
    return changes, context

# --- Mode 1: Impact Report (for pre-commit) ---
def run_impact_analysis():
    print(f"📂 Repository: {REPO_PATH}")
    changes, context = get_diff_and_context()
    
    if not changes:
        print("✨ No code changes to analyze.")
        return

    print("🤖 Generating Impact Analysis...")
    prompt = f"""
    Analyze these staged changes:
    {list(changes.values())}
    Context:
    {context}
    Output: Summary, Risks (Low/Med/High), and Suggestions. concise.
    """
    try:
        client = ZhipuAI(api_key=API_KEY)
        res = client.chat.completions.create(
            model="glm-4-flash", messages=[{"role": "user", "content": prompt}]
        )
        print("\n" + "="*50 + "\n📊 AI IMPACT REPORT\n" + "="*50)
        print(res.choices[0].message.content)
        print("="*50 + "\n")
    except Exception as e:
        print(f"❌ Analysis failed: {e}")

# --- Mode 2: Commit Suggestion (for commit-msg) ---
def run_commit_suggestion(msg_file_path):
    # 1. 读取用户原始输入
    with open(msg_file_path, 'r', encoding='utf-8') as f:
        original_msg = f.read().strip()
    
    # 如果用户没写任何东西，或者已经是 Merge，跳过
    if not original_msg: return

    print(f"🔄 Analyzing commit message: '{original_msg}'...")
    changes, context = get_diff_and_context()
    
    if not changes: return

    # 2. 生成 3 个选项
    prompt = f"""
    User's draft commit message: "{original_msg}"
    
    Code Changes:
    {list(changes.values())}
    
    Task: Generate 3 distinct commit messages based on the code changes and user intent.
    Format requirements:
    - Option 1: Standard Conventional Commit (e.g., feat: add login).
    - Option 2: Detailed with bullet points.
    - Option 3: Use Emojis (e.g., ✨ Feature: ...).
    
    Output ONLY the 3 options, separated by '|||'. Do not output anything else.
    Example:
    feat: update logic|||fix: logic error\n- fixed null pointer|||🐛 Fix: logic
    """
    
    try:
        client = ZhipuAI(api_key=API_KEY)
        res = client.chat.completions.create(
            model="glm-4-flash", messages=[{"role": "user", "content": prompt}]
        )
        raw_options = res.choices[0].message.content.split('|||')
        options = [opt.strip() for opt in raw_options if opt.strip()]
        
        # 补齐 3 个以防万一
        while len(options) < 3: options.append("refactor: update code")

    except Exception as e:
        print(f"❌ Suggestion failed: {e}")
        return

    # 3. 交互式选择菜单
    print("\n" + "="*60)
    print("🤖 AI COMMIT SUGGESTIONS")
    print("="*60)
    print(f"0️⃣  [Keep Original]: {original_msg}")
    print(f"1️⃣  {options[0]}")
    print(f"2️⃣  {options[1]}")
    print(f"3️⃣  {options[2]}")
    print("="*60)

    # 4. 获取用户选择 (使用强制 TTY 输入)
    selection = get_console_input("\n👉 Select an option (0-3) [Enter for 0]: ")

    final_msg = original_msg
    if selection == '1':
        final_msg = options[0]
    elif selection == '2':
        final_msg = options[1]
    elif selection == '3':
        final_msg = options[2]
    
    # 5. 覆写文件
    if final_msg != original_msg:
        with open(msg_file_path, 'w', encoding='utf-8') as f:
            f.write(final_msg)
        print(f"✅ Message updated to: {final_msg[:50]}...")
    else:
        print("👌 Keeping original message.")

if __name__ == "__main__":
    # 如果有参数传入 (文件名)，说明是 commit-msg 钩子在调用
    if len(sys.argv) > 1:
        run_commit_suggestion(sys.argv[1])
    else:
        # 如果没有参数，说明是 pre-commit 钩子在调用
        run_impact_analysis()