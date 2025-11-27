#!/bin/sh

# --- 配置部分 ---
# 如果你的 python 命令是 python3，请修改这里
# 如果你使用了虚拟环境 (venv)，建议写 Python解释器的绝对路径
# 例如 Windows: PYTHON_EXEC="./venv/Scripts/python.exe"
# 例如 Mac/Linux: PYTHON_EXEC="./venv/bin/python"
PYTHON_EXEC="python"

# 假设 analyzer.py 在项目根目录。如果在子目录，请修改路径，如 "src/analyzer.py"
SCRIPT_PATH="git_hook/analyzer.py"

# --- UI 展示 ---
echo ""
echo "========================================================"
echo "🤖 正在启动 AI 代码变更影响分析 (RAG + ZhipuAI)..."
echo "========================================================"

# --- 执行分析器 ---
# 这里的 $? 是获取上一个命令的退出状态码
$PYTHON_EXEC $SCRIPT_PATH

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "❌ 分析脚本运行出错，或检测到严重风险。"
    echo "   (如果你想忽略此错误强制提交，请使用 git commit --no-verify)"
    echo "========================================================"
    
    # 如果你想在分析出错时阻止提交，请取消下面这行的注释：
    # exit 1
else
    echo ""
    echo "✅ 分析完成。"
    echo "========================================================"
fi

# 退出码 0 表示允许提交继续
exit 0