# File: client/git_guard_cli.py
import os
import requests
import stat

SERVER_URL = "http://localhost:8000"

# ==============================================================================
# 钩子脚本模板 (Shell Scripts)
# ==============================================================================

# 1. Commit-Msg Hook: 负责【读取】和建议
# 触发时机：git commit 时
# 行为：前台运行，阻塞式（必须等待用户交互选择消息）
HOOK_COMMIT_MSG = """#!/bin/sh
PYTHON_EXEC="python"
GIT_DIR=$(git rev-parse --git-dir)
SCRIPT="$GIT_DIR/hooks/git_guard_analyzer.py"

# $1 是 commit message 临时文件路径
"$PYTHON_EXEC" "$SCRIPT" "$1"

# 退出码决定是否允许提交
exit $?
"""

# 2. Pre-Push Hook: 负责【写入】和更新索引
# 触发时机：git push 时
# 行为：后台异步运行 (Asynchronous)，完全不阻塞 Git Push 过程
HOOK_PRE_PUSH = """#!/bin/sh
PYTHON_EXEC="python"
GIT_DIR=$(git rev-parse --git-dir)
SCRIPT="$GIT_DIR/hooks/git_guard_indexer.py"
LOG_FILE="$GIT_DIR/../indexer_debug.log" # 将日志输出到项目根目录

echo "------------------------------------------------"
echo "🚀 Git-Guard: Triggering Knowledge Base Update..."

# [调试模式] 输出重定向到 log 文件，而不是 /dev/null
"$PYTHON_EXEC" "$SCRIPT" > "$LOG_FILE" 2>&1 &

echo "✅ Background indexing started."
echo "   (Check indexer_debug.log for details)"
echo "------------------------------------------------"

exit 0
"""

# ==============================================================================
# 安装逻辑
# ==============================================================================

def download_script(script_type, save_path):
    """从 Server 下载最新的 Python 逻辑"""
    try:
        print(f"☁️  Fetching {script_type} logic from cloud...")
        url = f"{SERVER_URL}/api/v1/scripts/{script_type}"
        resp = requests.get(url, timeout=5)
        
        if resp.status_code == 200:
            content = resp.json().get("code")
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"✅ Installed: {save_path}")
            return True
        else:
            print(f"❌ Server Error ({resp.status_code}): Could not fetch {script_type}")
            return False
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False

def install():
    print(f"🔧 Git-Guard Installer v3.0 (Commit-Read / Push-Write Mode)")
    print(f"   Target Server: {SERVER_URL}")
    print("-" * 30)

    if not os.path.exists(".git"):
        print("❌ Error: Not a git repository.")
        return

    hooks_dir = os.path.join(".git", "hooks")
    if not os.path.exists(hooks_dir): os.makedirs(hooks_dir)

    # --- 步骤 1: 下载 Python 核心脚本 ---
    
    # 1.1 下载 Analyzer (用于 commit 建议)
    analyzer_dest = os.path.join(hooks_dir, "git_guard_analyzer.py")
    if not download_script("analyzer", analyzer_dest): return

    # 1.2 下载 Indexer (用于 push 更新)
    indexer_dest = os.path.join(hooks_dir, "git_guard_indexer.py")
    if not download_script("indexer", indexer_dest): return

    # --- 步骤 2: 配置 Git Hooks ---

    # 2.1 配置 commit-msg
    c_path = os.path.join(hooks_dir, "commit-msg")
    with open(c_path, "w", encoding="utf-8") as f: f.write(HOOK_COMMIT_MSG)
    os.chmod(c_path, os.stat(c_path).st_mode | stat.S_IEXEC)
    print(f"✅ Hook 'commit-msg' configured (Trigger: git commit).")

    # 2.2 配置 pre-push
    p_path = os.path.join(hooks_dir, "pre-push")
    with open(p_path, "w", encoding="utf-8") as f: f.write(HOOK_PRE_PUSH)
    os.chmod(p_path, os.stat(p_path).st_mode | stat.S_IEXEC)
    print(f"✅ Hook 'pre-push' configured (Trigger: git push).")

    # --- 步骤 3: 清理旧钩子 (防止冲突) ---
    # 如果用户之前安装过 post-commit 版本，删掉它
    old_hook = os.path.join(hooks_dir, "post-commit")
    if os.path.exists(old_hook):
        os.remove(old_hook)
        print("🗑️  Cleaned up legacy 'post-commit' hook.")

    print("-" * 30)
    print("🚀 Installation Complete!")
    print("   1. Run 'git commit': AI analyzes code & suggests messages (Read-Only).")
    print("   2. Run 'git push':   AI updates local vector database (Write).")

if __name__ == "__main__":
    install()