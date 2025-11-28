# -*- coding: utf-8 -*-
"""
git_guard_collect_from_cases_to_ragas_json.py

作用：
  - 读取 git_guard_eval_cases.json 中你已经准备好的评测样本
    每条样本格式为：
      {
        "user_input": "...",
        "response": "",
        "retrieved_contexts": ["...", "..."],
        "reference": "..."
      }

  - 把 user_input 作为问题，批量发送到正在运行的 RAG 系统
  - 从 RAG 返回里提取回答（response）和检索到的上下文（retrieved_contexts）
  - 生成 ragas_eval_from_manual.json，结构可直接被 ragas_eval_from_manual.py 使用
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Tuple

import requests

BASE_DIR = os.path.dirname(__file__)

CASES_PATH = os.path.join(BASE_DIR, "git_guard_eval_cases.json")
OUTPUT_JSON_PATH = os.path.join(BASE_DIR, "ragas_eval_from_manual.json")

# 你自己的 RAG 后端地址（需要按实际情况修改）
# 例如：FastAPI 跑在 8000 端口： http://localhost:8000/api/rag/query
RAG_ENDPOINT = "http://localhost:8000/api/rag/query"

# 每次请求之间的间隔，避免打爆后端
SLEEP_SECONDS = 0.5


def load_cases(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到评测样例文件: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("git_guard_eval_cases.json 顶层必须是一个 list")
    return data


def build_payload(question: str) -> Dict[str, Any]:
    """
    根据你自己后端的接口格式构造请求体。

    ⚠️ 一定要结合你前端实际调用的接口来改：
      - 打开前端页面（如果有），按 F12 → Network
      - 找到你问问题时发出去的那个请求
      - 看请求 URL 和 body 结构，照着改

    下面是一个最简单的示例：后端希望收到 {"query": "..."}：
    """
    return {"query": question}


def extract_answer_and_contexts(resp_json: Any) -> Tuple[str, List[str]]:
    """
    从 RAG 返回的 JSON 里提取：
      - answer: 模型最终回答
      - contexts: 一个字符串列表，包含检索到的文档片段 / evidence

    ⚠️ 这里的实现是“通用兜底版”，你需要根据自己后端真实返回结构做适配。
    """
    answer = ""
    contexts: List[str] = []

    if not isinstance(resp_json, dict):
        return str(resp_json), []

    # 1) 尝试常见的答案字段
    for key in ["answer", "response", "result", "output", "message"]:
        if key in resp_json and isinstance(resp_json[key], str):
            answer = resp_json[key].strip()
            break

    # 2) 尝试从嵌套字段里找，比如 {"data": {"answer": "..."}}
    if not answer and isinstance(resp_json.get("data"), dict):
        data = resp_json["data"]
        for key in ["answer", "response", "result"]:
            if key in data and isinstance(data[key], str):
                answer = data[key].strip()
                break

    # 3) 尝试提取上下文列表
    for key in ["contexts", "retrieved_contexts", "docs", "documents", "evidences"]:
        raw_ctxs = resp_json.get(key)
        if not raw_ctxs:
            continue
        if isinstance(raw_ctxs, list):
            for c in raw_ctxs:
                if isinstance(c, str):
                    contexts.append(c)
                elif isinstance(c, dict):
                    text = (
                        c.get("text")
                        or c.get("content")
                        or c.get("answer")
                        or json.dumps(c, ensure_ascii=False)
                    )
                    contexts.append(str(text))
                else:
                    contexts.append(str(c))
            break

    return answer, contexts


def call_rag_backend(question: str) -> Tuple[str, List[str]]:
    """
    把一个问题发送到 RAG 系统，并解析返回的答案和上下文。
    """
    payload = build_payload(question)

    try:
        resp = requests.post(RAG_ENDPOINT, json=payload, timeout=60)
    except Exception as e:
        print(f"[ERROR] 请求 RAG 后端失败: {e}")
        return "", []

    if resp.status_code != 200:
        print(f"[ERROR] RAG 返回 HTTP {resp.status_code}: {resp.text[:200]}")
        return "", []

    try:
        resp_json = resp.json()
    except Exception as e:
        print(f"[ERROR] 解析 RAG JSON 失败: {e}, 原始文本前 200 字符: {resp.text[:200]}")
        return "", []

    answer, contexts = extract_answer_and_contexts(resp_json)
    return answer, contexts


def main() -> None:
    print("=== Git-Guard: 使用 git_guard_eval_cases.json 调用 RAG，生成 RAGAS 评估输入 ===")

    cases = load_cases(CASES_PATH)
    print(f"[INFO] 共读取 {len(cases)} 条样本。")

    ragas_samples: List[Dict[str, Any]] = []

    for idx, case in enumerate(cases, start=1):
        user_input = str(case.get("user_input", "")).strip()
        reference = str(case.get("reference", "")).strip()
        base_contexts = case.get("retrieved_contexts") or []

        if not user_input:
            print(f"[WARN] 第 {idx} 条样本缺少 user_input，跳过。")
            continue
        if not reference:
            print(f"[WARN] 第 {idx} 条样本缺少 reference，建议补全标准答案。")

        print(f"\n[CASE {idx}] 提问: {user_input}")
        answer, rag_contexts = call_rag_backend(user_input)

        # 如果 RAG 没有返回上下文，就用 git_guard_eval_cases.json 里自带的
        contexts = rag_contexts if rag_contexts else base_contexts

        sample = {
            "user_input": user_input,
            "response": answer,
            "retrieved_contexts": contexts,
            "reference": reference,
        }
        ragas_samples.append(sample)

        time.sleep(SLEEP_SECONDS)

    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(ragas_samples, f, ensure_ascii=False, indent=2)

    print(f"\n[DONE] 已生成 RAGAS 输入文件: {OUTPUT_JSON_PATH}")
    print("      结构：user_input, response, retrieved_contexts, reference")
    print("      接下来可直接运行: python ragas_eval_from_manual.py")


if __name__ == "__main__":
    main()
