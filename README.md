# Git-Guard: AI-Driven Distributed DevSecOps Platform

**Git-Guard** 是一个分布式的、基于大语言模型（LLM）的智能代码提交辅助与质量监控平台。它利用 **Hybrid RAG (混合检索增强生成)** 和 **Semantic Rerank (语义重排序)** 技术，将代码规范检查、提交信息生成、风险评估与 CI/CD 流水线深度集成，旨在解决团队开发中规范不统一、代码审查滞后和工具配置繁琐等痛点。


## Key Features

### 1\. Intelligent Commit Assistant 

  * **Context-Aware Suggestions**: 利用 **ChromaDB** 本地向量库，结合 **Hybrid Retrieval (Vector + Keyword)** 和 **Rerank** 算法，分析当前 `git diff` 与历史代码的语义关联。
  * **Auto-Generation**: 根据团队定义的模板（如 `[Module][Type] Description`），自动生成 3 个符合规范的 Commit Message 建议。
  * **Security Gate**: 在 `pre-commit` 阶段自动拦截硬编码密码、Token 泄露等高风险代码，并提供修复建议。

### 2\. Centralized Configuration Management

  * **Dynamic Rule Distribution**: Team Leader 可在云端 Dashboard 修改提交规范（Prompt 模板），所有客户端下次提交时**自动热更新**，无需重新分发脚本。
  * **Commit Tracking**: 实时收集全团队的提交日志、风险等级和 AI 分析摘要，实现项目进度的可视化监控。

### 3\. Automated CI/CD & Knowledge Sync 

  * **Asynchronous Processing**: 利用 `pre-push` 钩子在后台异步触发 Indexer，确保不阻塞开发者的 Push 操作。
  * **Sandboxed CI Environment**: 服务器端内置 Cronjob，定期拉取代码并在沙箱环境中运行全量测试 (`pytest`)。
  * **Self-Healing**: 具备 Reset & Clean 机制，确保 CI 环境的一致性。

### 4\. Visual Management Dashboard 

  * 基于 **Vue 3 + Tailwind CSS** 的现代化管理后台。
  * 支持移动端适配，随时随地查看 CI 状态和修改团队规则。

-----

## Architecture

Git-Guard 采用 **Client-Server (C/S)** 分离架构，兼顾了本地执行的低延迟与云端管理的统一性。

```mermaid
graph TD
    User[Developer] -->|git commit| Hook1[Pre-Commit Hook]
    User -->|git push| Hook2[Pre-Push Hook]
    
    subgraph Client [Local Environment]
        Hook1 --> Analyzer[Analyzer Script]
        Hook2 --> Indexer[Indexer Script]
        Analyzer <--> LocalDB[(Local ChromaDB)]
        Indexer -->|Update| LocalDB
    end
    
    subgraph Server [Cloud Platform]
        API[FastAPI Server]
        Dashboard[Vue3 Frontend]
        Scheduler[APScheduler CI]
        LogDB[(Commit Logs)]
    end
    
    Analyzer -->|Fetch Rules| API
    Analyzer -->|Report Logs| API
    API <--> LogDB
    Scheduler -->|Run Tests| CI_Env[CI Sandbox]
```

-----

## Quick Start

### Prerequisites

  * Python 3.10+
  * Git
  * Docker & Docker Compose (Optional for server deployment)
  * **ZhipuAI API Key** (Set as `ZHIPU_API_KEY` environment variable)

### 🛠️ Server Deployment 

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-repo/git-guard.git
    cd git-guard
    ```

2.  **Run with Docker Compose (Recommended):**

    ```bash
    # Set your API Key
    export ZHIPU_API_KEY="your_api_key_here"

    # Start Backend & Frontend
    docker-compose up --build -d
    ```

      * **Dashboard:** `http://localhost` (or Server IP)
      * **API:** `http://localhost:8000`

### Client Installation 

开发者只需运行一条命令即可完成环境初始化（自动安装依赖、下载钩子脚本、初始化本地向量库）：

1.  **Configure Server IP:**
    Edit `client/git_guard_cli.py` and set `SERVER_URL` to your server's address.

2.  **Run Installer:**

    ```bash
    cd your-project-root
    cp this_project/client/git_guard_cli.py /path/to/your/project/root/
    python /path/to/your/project/root/git_guard_cli.py
    # then you can delete git_guard_cli.py
    ```

3.  **That's it\!** Now try `git commit -m "test"` to see the AI magic. 




## Demo Scenario

1.  **Safety Guard**: Try committing code with `password = "123"`. Git-Guard will intercept and warn about security risks.
2.  **AI Suggestion**: Fix a bug and commit. Git-Guard analyzes the `diff`, retrieves related context, and suggests a standard message like `[Backend][Fix] resolve login timeout`.
3.  **Dynamic Config**: Change the rule to "Use Emojis" on the Dashboard. The next commit suggestion immediately reflects this change.
4.  **Auto CI**: Trigger the pipeline. Watch the server verify the codebase automatically.


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.