# File: server/main.py
import os
import json
import csv
import shutil
import subprocess
import asyncio
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks 
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from typing import Optional, Dict
from git import Repo
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

# ==========================================
# Config: File Paths
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE_PATH = os.path.join(BASE_DIR, "server_config.json")
LOG_FILE_PATH = os.path.join(BASE_DIR, "commit_history.csv")
CI_STATUS_PATH = os.path.join(BASE_DIR, "ci_status.json")
CI_WORKSPACE_DIR = os.path.join(BASE_DIR, "ci_workspace")

# ==========================================
# Config: Default Settings
# ==========================================
DEFAULT_CONFIG = {
    "template_format": "[<Module>][<Type>] <Description>",
    "custom_rules": "1. <Module>: [Backend], [Frontend]. 2. <Type>: [Feat], [Fix].",
    "github_repo_url": "", 
    "ci_interval_minutes": 60
}

# ==========================================
# Global: Scheduler
# ==========================================
scheduler = AsyncIOScheduler()

# ==========================================
# Helper Functions: Persistence
# ==========================================
def load_config_from_disk() -> dict:
    if not os.path.exists(CONFIG_FILE_PATH): return DEFAULT_CONFIG
    try:
        with open(CONFIG_FILE_PATH, 'r', encoding='utf-8') as f:
            config = json.load(f)
            # Ensure new keys exist for backward compatibility
            for k, v in DEFAULT_CONFIG.items():
                if k not in config: config[k] = v
            return config
    except: return DEFAULT_CONFIG

def save_config_to_disk(config_data: dict):
    try:
        with open(CONFIG_FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, ensure_ascii=False, indent=4)
        # Reschedule CI job if config updated
        reschedule_ci_job(config_data.get("ci_interval_minutes", 60))
    except Exception as e:
        print(f"❌ Failed to save config: {e}")

def save_log_to_csv(log):
    file_exists = os.path.exists(LOG_FILE_PATH)
    try:
        with open(LOG_FILE_PATH, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Timestamp", "Developer", "Repo", "Risk", "Message", "AI Summary"])
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow([timestamp, log.developer_id, log.repo_name, log.risk_level, log.commit_msg, log.ai_summary])
    except: pass

# CI Status Management
def load_ci_status():
    if not os.path.exists(CI_STATUS_PATH):
        return {"status": "Never Ran", "last_run": None, "details": "No logs yet."}
    try:
        with open(CI_STATUS_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except: return {"status": "Error", "details": "File read error"}

def save_ci_status(status, details):
    data = {
        "last_run": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": status,
        "details": details 
    }
    with open(CI_STATUS_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# ==========================================
# Core Logic: CI Task (Pull & Test)
# ==========================================
def run_ci_task():
    print("\n⏰ [CI Job] Starting scheduled CI task...")
    config = load_config_from_disk()
    repo_url = config.get("github_repo_url")

    if not repo_url:
        print("⚠️ [CI Job] No 'github_repo_url' configured. Skipping.")
        save_ci_status("Skipped", "Repo URL not configured.")
        return

    try:
        # 1. Setup workspace
        if not os.path.exists(CI_WORKSPACE_DIR):
            os.makedirs(CI_WORKSPACE_DIR)
            print(f"   Cloning {repo_url}...")
            Repo.clone_from(repo_url, CI_WORKSPACE_DIR)
        else:
            try:
                repo = Repo(CI_WORKSPACE_DIR)
                print("   Pulling latest code (main)...")
                repo.git.checkout('main')
                repo.remotes.origin.pull()
            except Exception as e:
                # Force re-clone on git error
                print(f"   Git pull failed ({e}), re-cloning...")
                shutil.rmtree(CI_WORKSPACE_DIR)
                os.makedirs(CI_WORKSPACE_DIR)
                Repo.clone_from(repo_url, CI_WORKSPACE_DIR)

        # 2. Run Pytest
        print("   Running Pytest...")
        result = subprocess.run(
            ["pytest"], 
            cwd=CI_WORKSPACE_DIR, 
            capture_output=True, 
            text=True,
            shell=True # Required for Windows
        )

        # 3. Log results
        output_log = result.stdout + "\n" + result.stderr
        if result.returncode == 0:
            print("✅ [CI Job] Tests Passed!")
            save_ci_status("Success", output_log)
        else:
            print("❌ [CI Job] Tests Failed.")
            save_ci_status("Failed", output_log)

    except Exception as e:
        print(f"❌ [CI Job] System Error: {e}")
        save_ci_status("System Error", str(e))

def reschedule_ci_job(interval_minutes):
    """Update job interval"""
    try:
        scheduler.remove_all_jobs()
        scheduler.add_job(
            run_ci_task, 
            IntervalTrigger(minutes=max(1, interval_minutes)),
            id="ci_job",
            replace_existing=True
        )
        print(f"🔄 CI Job Rescheduled: Every {interval_minutes} minutes.")
    except Exception as e:
        print(f"⚠️ Scheduler Error: {e}")

# ==========================================
# Lifespan: Scheduler Management
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    config = load_config_from_disk()
    interval = config.get("ci_interval_minutes", 60)
    
    scheduler.add_job(run_ci_task, IntervalTrigger(minutes=interval), id="ci_job")
    scheduler.start()
    print("🚀 Scheduler Started.")
    
    yield
    
    # Shutdown
    scheduler.shutdown()
    print("🛑 Scheduler Shutdown.")

# ==========================================
# App Initialization
# ==========================================
app = FastAPI(title="Git-Guard Cloud Server", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CommitLog(BaseModel):
    developer_id: str
    repo_name: str
    commit_msg: str
    risk_level: str
    ai_summary: str

class ProjectConfig(BaseModel):
    template_format: str
    custom_rules: str
    github_repo_url: Optional[str] = ""
    ci_interval_minutes: Optional[int] = 60

# ==========================================
# API Endpoints
# ==========================================

@app.get("/api/v1/scripts/{script_name}")
def get_script(script_name: str):
    valid_scripts = {"analyzer": "analyzer_template.py", "indexer": "indexer_template.py"}
    if script_name not in valid_scripts: raise HTTPException(status_code=404)
    file_path = os.path.join(BASE_DIR, valid_scripts[script_name])
    if not os.path.exists(file_path): raise HTTPException(status_code=500)
    with open(file_path, "r", encoding="utf-8") as f: return {"code": f.read()}

@app.post("/api/v1/track")
def track_commit(log: CommitLog):
    print(f"📡 [TRACKING] {log.developer_id}: {log.commit_msg}")
    save_log_to_csv(log)
    return {"status": "recorded"}

@app.post("/api/v1/config")
def update_config(config: ProjectConfig):
    new_config = config.dict()
    save_config_to_disk(new_config) # Auto reschedules job
    print(f"⚙️  Config Updated: {new_config}")
    return {"status": "updated", "config": new_config}

@app.get("/api/v1/config")
def get_config():
    return load_config_from_disk()

@app.get("/api/v1/ci/status")
def get_ci_status():
    return load_ci_status()

@app.post("/api/v1/ci/run")
def trigger_ci_manually(background_tasks: BackgroundTasks = None): 
    job = scheduler.get_job("ci_job")
    if job:
        job.modify(next_run_time=datetime.now()) # Execute immediately
        return {"status": "Triggered"}
    return {"status": "Error", "details": "Job not found"}

if __name__ == "__main__":
    print(f"🚀 Server Starting...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, reload_dirs=[BASE_DIR])