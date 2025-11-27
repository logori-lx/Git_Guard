#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
merge_numbered_manual_to_ragas_json.py

功能：
  1. 读取你已经按「1.问题」「2.问题」编号好的：
       新建 文本文档.txt
     每个块结构类似：
       1.我有高血压很多年了，最近经常头晕、有点胸闷，想问平时在生活方式和复查方面需要注意什么？
       1) ...
       2) ...
       ...
       隐藏参考案例 Reference cases ▲
       参考案例（Reference cases）
       Case 1
       Question: ...
       Answer: ...
       科室：...
       ...
  2. 读取标准答案：reference_answers.json
     - 结构类似：[{"id": 1, "reference": "..."}, ...]
  3. 按题目前面的编号（1. / 2. / …）对齐到对应 id 的 reference。
  4. 输出 ragas_eval_from_manual.json，格式严格为：
        {
          "user_input": "...",
          "response": "...",
          "retrieved_contexts": ["{json_doc_1}", "{json_doc_2}", ...],
          "reference": "..."
        }

用法：
    python merge_numbered_manual_to_ragas_json.py
"""

import json
import os
import re
from typing import Any, Dict, List

# ====== 文件名配置 ======
MANUAL_FILE = "新建 文本文档.txt"          # 你编号后的文本
REFERENCE_FILE = "reference_answers.json"   # 标准答案
OUTPUT_FILE = "ragas_eval_from_manual.json" # 输出给 ragas 用的 json


# ---------- 基础 I/O ----------

def load_text(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到文件：{path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_references(path: str) -> Dict[int, str]:
    """
    从 reference_answers.json 中加载 {id -> reference} 映射。

    允许两种结构：
      1) [{"id": 1, "reference": "..."}, ...]
      2) ["...", "...."]  -> 自动给 id = 1..N
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到标准答案文件：{path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    refs_by_id: Dict[int, str] = {}

    if isinstance(data, list):
        # 元素是 dict 或 str
        for idx, item in enumerate(data):
            if isinstance(item, dict):
                # 要求有 reference 字段
                if "reference" not in item:
                    raise ValueError(
                        f"{path} 中第 {idx} 条缺少 'reference' 字段：{item}"
                    )
                rid = int(item.get("id", idx + 1))
                refs_by_id[rid] = str(item["reference"])
            else:
                # 直接是字符串
                refs_by_id[idx + 1] = str(item)
    else:
        raise ValueError(f"{path} 顶层 JSON 必须是 list，当前是：{type(data)}")

    return refs_by_id


# ---------- 解析上下文（参考案例） ----------

def parse_contexts_from_block(ctx_lines: List[str]) -> List[str]:
    """
    解析一个问题块中「参考案例」部分，生成 TopK retrieved_contexts。

    ctx_lines：从“参考案例（Reference cases）”下一行开始的所有行。

    返回：
        list[str]，每个元素是一个 JSON 字符串：
        {"id": 1, "metadata": {"department": "...", "question": "..."}, "text": "..."}
    """
    cases: List[Dict[str, Any]] = []
    current: Dict[str, Any] | None = None
    answer_buf: List[str] = []

    for raw in ctx_lines:
        line = raw.strip()
        if not line:
            continue

        # Case N
        if line.startswith("Case "):
            # 收尾上一个 case
            if current is not None:
                current["answer"] = "".join(answer_buf).strip()
                cases.append(current)
            # 开新 case
            m = re.match(r"Case\s+(\d+)", line)
            cid = int(m.group(1)) if m else (len(cases) + 1)
            current = {
                "id": cid,
                "question": "",
                "answer": "",
                "department": "",
            }
            answer_buf = []
            continue

        # Question: ...
        if line.startswith("Question:"):
            if current is None:
                current = {
                    "id": len(cases) + 1,
                    "question": "",
                    "answer": "",
                    "department": "",
                }
                answer_buf = []
            current["question"] = line.split("Question:", 1)[1].strip()
            continue

        # Answer: ...
        if line.startswith("Answer:"):
            if current is None:
                current = {
                    "id": len(cases) + 1,
                    "question": "",
                    "answer": "",
                    "department": "",
                }
                answer_buf = []
            answer_buf.append(line.split("Answer:", 1)[1].strip() + " ")
            continue

        # 科室：...
        if line.startswith("科室"):
            if current is None:
                current = {
                    "id": len(cases) + 1,
                    "question": "",
                    "answer": "",
                    "department": "",
                }
            # 兼容 “科室：” 和 “科室:”
            if "：" in line:
                dep = line.split("：", 1)[1].strip()
            elif ":" in line:
                dep = line.split(":", 1)[1].strip()
            else:
                dep = line
            current["department"] = dep
            continue

        # 其他行：如果 answer_buf 已经开始，则视作答案的续行
        if answer_buf:
            answer_buf.append(line + " ")

    # 收尾最后一个 case
    if current is not None:
        current["answer"] = "".join(answer_buf).strip()
        cases.append(current)

    # 转为 JSON 字符串形式
    ctx_json_list: List[str] = []
    for c in cases:
        meta = {
            "department": c.get("department", ""),
            "question": c.get("question", ""),
        }
        doc = {
            "id": c.get("id", 0),
            "metadata": meta,
            "text": c.get("answer", ""),
        }
        ctx_json_list.append(json.dumps(doc, ensure_ascii=False))

    # 一般是 Top5，这里做一个防御性截断
    return ctx_json_list[:5]


# ---------- 解析整份编号后的 txt ----------

def parse_manual_numbered_txt(text: str) -> List[Dict[str, Any]]:
    """
    将编号后的 新建 文本文档.txt 解析成样本列表。

    约定：每个问题块以 “数字.问题” 开头，例如：
        1.我有高血压很多年了，最近经常头晕、有点胸闷，想问平时在生活方式和复查方面需要注意什么？
        1) ...
        ...
        隐藏参考案例 Reference cases ▲
        参考案例（Reference cases）
        Case 1
        ...

    返回的每条样本包含：
        {
          "id": 1,
          "user_input": "...",
          "response": "...",
          "retrieved_contexts": [json_str, ...]
        }
    """
    lines = text.splitlines()

    # 找到所有形如 “数字.问题” 的行号
    q_indices: List[int] = []
    q_ids: List[int] = []
    q_texts: List[str] = []

    pattern = re.compile(r"^\s*(\d+)\.(.+)$")

    for i, line in enumerate(lines):
        m = pattern.match(line)
        if m:
            q_indices.append(i)
            q_ids.append(int(m.group(1)))
            q_texts.append(m.group(2).strip())

    if not q_indices:
        raise ValueError("没有检测到形如 '1.问题内容' 的行，请确认 txt 已按编号格式整理。")

    samples: List[Dict[str, Any]] = []

    for idx, start in enumerate(q_indices):
        end = q_indices[idx + 1] if idx + 1 < len(q_indices) else len(lines)
        block_lines = lines[start:end]

        qid = q_ids[idx]
        user_input = q_texts[idx]

        # 找回答和参考案例的分界：
        #   回答：从问题下一行开始，到 “隐藏参考案例 Reference cases” 或
        #         “参考案例（Reference cases）” 之前。
        #   上下文：从 “参考案例（Reference cases）” 下一行开始。
        idx_hide = None
        idx_ref = None
        for j, l in enumerate(block_lines):
            if "隐藏参考案例" in l:
                idx_hide = j
            if "参考案例（Reference cases）" in l:
                idx_ref = j
                break  # 第一次出现即可

        # 回答部分
        ans_start = 1  # 从问题下一行开始
        if idx_hide is not None:
            ans_end = idx_hide
        elif idx_ref is not None:
            ans_end = idx_ref
        else:
            ans_end = len(block_lines)

        answer_lines = [l for l in block_lines[ans_start:ans_end]]
        # 去掉前后空行
        while answer_lines and not answer_lines[0].strip():
            answer_lines.pop(0)
        while answer_lines and not answer_lines[-1].strip():
            answer_lines.pop()
        response = "\n".join(answer_lines).strip()

        # 上下文部分
        if idx_ref is not None:
            ctx_lines = block_lines[idx_ref + 1:]
            retrieved_contexts = parse_contexts_from_block(ctx_lines)
        else:
            retrieved_contexts = []

        samples.append(
            {
                "id": qid,
                "user_input": user_input,
                "response": response,
                "retrieved_contexts": retrieved_contexts,
            }
        )

    return samples


# ---------- 合并标准答案并输出 RAGAS JSON ----------

def merge_samples_with_references(
    samples: List[Dict[str, Any]],
    refs_by_id: Dict[int, str],
) -> List[Dict[str, Any]]:
    """
    根据样本里的 id，从 refs_by_id 中找对应 reference，
    生成 RAGAS 所需结构：
      {
        "user_input": ...,
        "response": ...,
        "retrieved_contexts": [...],
        "reference": ...
      }
    """
    merged: List[Dict[str, Any]] = []

    for s in samples:
        qid = int(s["id"])
        ref = refs_by_id.get(qid)
        if ref is None:
            print(f"[WARN] 在 reference_answers.json 中找不到 id={qid} 的标准答案，已跳过该题。")
            continue

        merged.append(
            {
                "user_input": s["user_input"],
                "response": s["response"],
                "retrieved_contexts": s["retrieved_contexts"],
                "reference": ref,
            }
        )

    return merged


def save_json(data: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    print("=== Merge numbered manual RAG file + reference_answers -> RAGAS JSON ===\n")

    # 1. 读取编号后的 txt
    try:
        txt = load_text(MANUAL_FILE)
        print(f"[INFO] 成功读取 {MANUAL_FILE}，长度 {len(txt)} 字符。")
    except Exception as e:
        print(f"[ERROR] 读取 {MANUAL_FILE} 失败：{e}")
        return

    # 2. 解析出 (id, user_input, response, retrieved_contexts)
    try:
        samples = parse_manual_numbered_txt(txt)
        print(f"[INFO] 解析得到 {len(samples)} 条样本。")
    except Exception as e:
        print(f"[ERROR] 解析编号文本失败：{e}")
        return

    # 3. 读取标准答案
    try:
        refs_by_id = load_references(REFERENCE_FILE)
        print(f"[INFO] 从 {REFERENCE_FILE} 读取到 {len(refs_by_id)} 条标准答案（按 id）。")
    except Exception as e:
        print(f"[ERROR] 读取标准答案失败：{e}")
        return

    # 4. 合并样本 + reference
    merged = merge_samples_with_references(samples, refs_by_id)
    print(f"[INFO] 合并后得到 {len(merged)} 条可用于 RAGAS 的样本。")

    # 5. 输出最终 JSON
    try:
        save_json(merged, OUTPUT_FILE)
        print(f"[OK] 已写出 RAGAS 输入文件：{OUTPUT_FILE}")
        print("     每条记录字段：user_input / response / retrieved_contexts / reference")
    except Exception as e:
        print(f"[ERROR] 写入 {OUTPUT_FILE} 失败：{e}")


if __name__ == "__main__":
    main()
