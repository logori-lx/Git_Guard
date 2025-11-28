# -*- coding: utf-8 -*-
"""
git_guard_eval_generate_ragas_input.py

作用：
  1. 读取 git_guard_eval_cases.json 中事先准备好的 Git-Guard 提交场景
  2. 调用 GLM-4（或你项目里已经封装好的 ZhipuAI 客户端）生成符合模板的 Commit Message 建议
  3. 把结果转换成 ragas_eval_from_manual.json，字段为：
        - user_input          -> scenario（场景自然语言描述）
        - response            -> 选中的 commit message（比如第 1 个）
        - retrieved_contexts  -> [diff, template_format, custom_rules]
        - reference           -> 人工写的标准 commit（reference）
  4. 后续直接运行 ragas_eval_from_manual.py 用 DeepSeek + RAGAS 自动打分
"""

from __future__ import annotations

import os
import json
import time
import re
from typing import Any, Dict, List

import requests

# ---------------- 基本路径配置 ----------------

BASE_DIR = os.path.dirname(__file__)
CASES_PATH = os.path.join(BASE_DIR, "git_guard_eval_cases.json")
RAGAS_INPUT_PATH = os.path.join(BASE_DIR, "ragas_eval_from_manual.json")

# ---------------- 大模型配置（ZhipuAI HTTP 示例） ----------------

ZHIPU_API_KEY = os.getenv("ZHIPUAI_API_KEY") or ""  # 建议通过环境变量提供
ZHIPU_API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
ZHIPU_MODEL = "glm-4"   # 根据你当前项目实际使用的模型名调整

REQUEST_SLEEP_SECONDS = 1.0   # 每条样本之间的间隔，避免 QPS 太高


def load_cases(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到评测样例文件: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("git_guard_eval_cases.json 顶层必须是一个 list")
    return data


def build_prompt_for_case(case: Dict[str, Any]) -> str:
    """
    尽量复用 server/analyzer_template.py 中的 prompt 结构，
    这里做了一个简化版，方便离线批量评估。
    """
    diff = case.get("diff", "")
    template_format = case.get("template_format", "[<Module>][<Type>] <Description>")
    custom_rules = case.get("custom_rules", "")

    prompt = f"""
Role: Senior Technical Lead conducting a Pre-commit Risk Assessment.

[INPUT DATA]
Code Changes (Diff):
{diff}

[MANDATORY CONFIGURATION]
1. Target Template: "{template_format}"
2. Custom Instructions: "{custom_rules}"

[TASK]
Based on the code changes and configuration, generate 3 distinct commit messages.

[STRICT OUTPUT FORMAT]
RISK: <Level>
SUMMARY: <Summary>
OPTIONS: <Msg1>|||<Msg2>|||<Msg3>

[CONSTRAINTS]
- Plain text only. NO Markdown.
- NO numbered lists (1. 2.).
- Use '|||' as the ONLY separator for OPTIONS.
""".strip()
    return prompt


def call_glm(prompt: str) -> str:
    """
    调用 ZhipuAI HTTP 接口。
    如果你项目里已经有 zai.ZhipuAiClient，可以把这里替换成你现有的封装。
    """
    if not ZHIPU_API_KEY:
        raise RuntimeError(
            "缺少 ZHIPUAI_API_KEY 环境变量，请先在系统环境变量中配置你的智谱 API Key。"
        )

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {ZHIPU_API_KEY}",
    }

    payload = {
        "model": ZHIPU_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are Git-Guard's commit assistant. "
                           "Follow the STRICT OUTPUT FORMAT exactly."
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "top_p": 0.9,
    }

    resp = requests.post(ZHIPU_API_URL, headers=headers, json=payload, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"ZhipuAI 调用失败: {resp.status_code} {resp.text}")

    data = resp.json()
    # 根据官方接口格式提取内容；如果你项目里实际字段不同，请对应修改
    try:
        content = data["choices"][0]["message"]["content"]
    except Exception as e:
        raise RuntimeError(f"解析 ZhipuAI 返回内容失败: {e}, 原始响应: {data}")
    return content


RISK_RE = re.compile(r"RISK\s*:\s*(.+)", re.IGNORECASE)
SUMMARY_RE = re.compile(r"SUMMARY\s*:\s*(.+)", re.IGNORECASE)
OPTIONS_RE = re.compile(r"OPTIONS\s*:\s*(.+)", re.IGNORECASE)


def parse_commit_response(text: str) -> Dict[str, Any]:
    """
    从模型的输出中解析出 Risk / Summary / Options。
    如果格式轻微偏差，也尽量兜底。
    """
    if not text:
        return {"risk": "", "summary": "", "options": [], "raw": ""}

    risk_match = RISK_RE.search(text)
    summary_match = SUMMARY_RE.search(text)
    options_match = OPTIONS_RE.search(text)

    risk = risk_match.group(1).strip() if risk_match else ""
    summary = summary_match.group(1).strip() if summary_match else ""
    options_raw = options_match.group(1).strip() if options_match else ""

    # 选项用 '|||' 分隔
    options: List[str] = []
    if options_raw:
        for part in options_raw.split("|||"):
            msg = part.strip()
            if msg:
                options.append(msg)

    return {
        "risk": risk,
        "summary": summary,
        "options": options,
        "raw": text,
    }


def main() -> None:
    print("=== Git-Guard: 生成 RAGAS 输入文件 (ragas_eval_from_manual.json) ===")

    cases = load_cases(CASES_PATH)
    print(f"[INFO] 共读取 {len(cases)} 条评测场景。")

    ragas_samples: List[Dict[str, Any]] = []

    for idx, case in enumerate(cases, start=1):
        cid = case.get("id", idx)
        scenario = str(case.get("scenario", "")).strip()
        diff = str(case.get("diff", "")).strip()
        template_format = str(case.get("template_format", "")).strip()
        custom_rules = str(case.get("custom_rules", "")).strip()
        reference = str(case.get("reference", "")).strip()

        if not scenario or not diff or not reference:
            print(f"[WARN] case id={cid} 缺少必要字段，跳过。")
            continue

        print(f"\n[CASE {cid}] {scenario}")
        prompt = build_prompt_for_case(case)

        try:
            raw_output = call_glm(prompt)
        except Exception as e:
            print(f"  [ERROR] 调用 GLM 失败：{e}")
            commit_msg = ""
            contexts = [diff, template_format, custom_rules]
        else:
            parsed = parse_commit_response(raw_output)
            options = parsed.get("options") or []
            # 简单策略：默认选第 1 个作为最终推荐 commit message
            commit_msg = options[0] if options else raw_output.strip()
            contexts = [
                diff,
                f"Template: {template_format}",
                f"Rules: {custom_rules}",
                f"LLM_SUMMARY: {parsed.get('summary', '')}",
            ]

        sample = {
            "user_input": scenario,     # 问题：场景描述
            "response": commit_msg,     # 模型给出的 commit message
            "retrieved_contexts": contexts,  # 这里直接把 diff + 规则当成上下文
            "reference": reference,     # 标准 commit
        }
        ragas_samples.append(sample)

        # 防止打爆接口
        time.sleep(REQUEST_SLEEP_SECONDS)

    # 写出 ragas_eval_from_manual.json
    with open(RAGAS_INPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(ragas_samples, f, ensure_ascii=False, indent=2)

    print(f"\n[DONE] 已写入 RAGAS 输入文件: {RAGAS_INPUT_PATH}")
    print("      每条样本包含字段: user_input, response, retrieved_contexts, reference")
    print("      接下来可直接运行: python ragas_eval_from_manual.py")


if __name__ == "__main__":
    main()
