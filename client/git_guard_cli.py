# File: client/git_guard_cli.py
import os
import sys
import subprocess
import requests
import stat

SERVER_URL = "http://localhost:8000"

# --- Hook 1: Commit-Msg ---
HOOK_COMMIT_MSG = """#!/bin/sh
PYTHON_EXEC="python"
GIT_DIR=$(git rev-parse --git-dir)
SCRIPT="$GIT_DIR/hooks/git_guard_analyzer.py"
"$PYTHON_EXEC" "$SCRIPT" "$1"
exit $?
"""

# --- Hook 2: Pre-Push ---
HOOK_PRE_PUSH = """#!/bin/sh
PYTHON_EXEC="python"
GIT_DIR=$(git rev-parse --git-dir)
SCRIPT="$GIT_DIR/hooks/git_guard_indexer.py"
LOG_FILE="$GIT_DIR/../indexer_debug.log"

echo "------------------------------------------------"
echo "🚀 Git-Guard: Triggering Knowledge Base Update..."
"$PYTHON_EXEC" "$SCRIPT" > "$LOG_FILE" 2>&1 &
echo "✅ Background indexing started."
echo "------------------------------------------------"
exit 0
"""

# --- Hook 3: Pre-Commit ---
HOOK_PRE_COMMIT = """#!/bin/sh
PYTHON_EXEC="python"
GIT_DIR=$(git rev-parse --git-dir)
SCRIPT="$GIT_DIR/hooks/git_guard_analyzer.py"
"$PYTHON_EXEC" "$SCRIPT"
exit $?
"""

# ==========================================
# 核心依赖列表 (Client 端运行所需)
# ==========================================
REQUIRED_PACKAGES = [
    "gitpython",
    "chromadb",
    "zai-sdk",
    "langchain-community",
    "langchain-text-splitters",
    "requests",
    "tiktoken"
]

def install_dependencies():
    """自动安装 Python 依赖"""
    print("\n📦 Checking & Installing Dependencies...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install"] + REQUIRED_PACKAGES
        )
        print("✅ Dependencies installed.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        print("   Please manually run: pip install " + " ".join(REQUIRED_PACKAGES))
        sys.exit(1)

def run_initial_indexing(indexer_path):
    """首次全量扫描"""
    print("\n🧠 Running Initial Codebase Indexing...")
    print("   (This may take a while for large repos)")
    try:
        subprocess.check_call([sys.executable, indexer_path])
        print("✅ Initial indexing complete. Knowledge base ready.")
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Initial indexing failed: {e}")
        print("   Don't worry, it will retry on your next 'git push'.")

def download_script(script_type, save_path):
    try:
        print(f"☁️  Fetching {script_type} logic...")
        url = f"{SERVER_URL}/api/v1/scripts/{script_type}"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(resp.json().get("code"))
            print(f"✅ Downloaded: {os.path.basename(save_path)}")
            return True
        else:
            print(f"❌ Server Error: {resp.status_code}")
            return False
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False

def install():
    print(f"🔧 Git-Guard Installer v4.0 (Auto-Setup)")
    print(f"   Target Server: {SERVER_URL}")
    print("-" * 30)

    if not os.path.exists(".git"):
        print("❌ Error: Not a git repo.")
        return

    hooks_dir = os.path.join(".git", "hooks")
    if not os.path.exists(hooks_dir): os.makedirs(hooks_dir)

    install_dependencies()

    analyzer_path = os.path.join(hooks_dir, "git_guard_analyzer.py")
    dl_1 = download_script("analyzer", analyzer_path)
    
    indexer_path = os.path.join(hooks_dir, "git_guard_indexer.py")
    dl_2 = download_script("indexer", indexer_path)

    if not (dl_1 and dl_2):
        print("\n⚠️  Script download failed. Aborting.")
        return

    run_initial_indexing(indexer_path)

    print("-" * 15 + " Configuring Hooks " + "-" * 15)

    def write_hook(filename, content):
        path = os.path.join(hooks_dir, filename)
        try:
            if os.path.exists(path):
                os.chmod(path, stat.S_IWRITE)
                os.remove(path)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            os.chmod(path, stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC)
            print(f"✅ Hook '{filename}' updated.")
        except Exception as e:
            print(f"❌ Failed to update '{filename}': {e}")

    # 4. 配置 Hooks
    write_hook("commit-msg", HOOK_COMMIT_MSG)
    write_hook("pre-push", HOOK_PRE_PUSH)
    write_hook("pre-commit", HOOK_PRE_COMMIT)

    # 5. 清理旧钩子
    old_hook = os.path.join(hooks_dir, "post-commit")
    if os.path.exists(old_hook):
        try: os.remove(old_hook)
        except: pass

    print("-" * 30)
    print("🚀 Installation & Setup Complete!")
    print("   Your repo is now guarded.")

if __name__ == "__main__":
    install()