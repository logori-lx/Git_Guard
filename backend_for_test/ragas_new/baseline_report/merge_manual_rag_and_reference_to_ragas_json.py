# merge_manual_and_reference_to_ragas_json.py
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List

# ================== 配置区域 ==================

# 你手动整理、编号好的文本文件（示例）：
# 1.问题
# 回答....
#
# 检索到的上下文：
#
# “ctx1...”
# “ctx2...”
MANUAL_TEXT_PATH = "新建 文本文档.txt"

# 我之前给你生成的标准答案 JSON
# 结构类似：
# [
#   {"user_input": "...", "reference": "..."},
#   ...
# ]
REFERENCE_JSON_PATH = "reference_answers.json"

# 输出：给 ragas 用的 JSON
RAGAS_OUTPUT_PATH = "ragas_eval_from_manual.json"

# =================================================


def parse_manual_text(path: str) -> List[Dict[str, Any]]:
    """
    解析你手动整理的文本，抽取：
      - user_input       : 问题
      - response         : RAG 系统回答
      - retrieved_contexts : list[str] 检索到的上下文
    文本格式示例（和你发的一样）：

    1.我有高血压很多年了...
    高血压患者出现头晕...

    検索到的上下文：

    “这是有高血压，一定要留意休息...”
    “高血压也应注意平日的养生保健...”
    “建议在积极运用药物控制血压...”

    2.我是2型糖尿病患者，血糖时高时低...
    ...
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]

    samples: List[Dict[str, Any]] = []
    current: Dict[str, Any] | None = None
    mode: str | None = None  # "answer" or "contexts"

    q_pattern = re.compile(r"^(\d+)[\.\、]\s*(.+)$")  # 匹配 "1.问题" / "1、问题"

    for raw_line in lines:
        line = raw_line.rstrip("\n")
        stripped = line.strip()

        # 检测新的题号开头
        m = q_pattern.match(stripped)
        if m:
            # 先收尾上一个样本
            if current is not None:
                response_text = "\n".join(current["answer_lines"]).strip()
                ctx_list = [c for c in current["contexts"] if c.strip()]
                samples.append(
                    {
                        "user_input": current["question"],
                        "response": response_text,
                        "retrieved_contexts": ctx_list,
                    }
                )

            qid = int(m.group(1))
            question = m.group(2).strip()
            current = {
                "id": qid,
                "question": question,
                "answer_lines": [],
                "contexts": [],
            }
            mode = "answer"
            continue

        # 还没进入任何题目前的杂项，直接跳过
        if current is None:
            continue

        # 检测“检索到的上下文：”切换模式
        if stripped.startswith("检索到的上下文"):
            mode = "contexts"
            continue

        # 空行处理：回答段落里保留空行，contexts 里丢弃空行
        if stripped == "":
            if mode == "answer":
                current["answer_lines"].append("")  # 保留段落空行
            # mode == "contexts" 时，空行就跳过
            continue

        # 根据当前模式写入不同区域
        if mode == "answer":
            # 保留原行（不 strip，防止丢失缩进/列表格式）
            current["answer_lines"].append(line)
        elif mode == "contexts":
            # 每一非空行做一个 context
            current["contexts"].append(stripped)

    # 文件结束时，把最后一个样本收尾
    if current is not None:
        response_text = "\n".join(current["answer_lines"]).strip()
        ctx_list = [c for c in current["contexts"] if c.strip()]
        samples.append(
            {
                "user_input": current["question"],
                "response": response_text,
                "retrieved_contexts": ctx_list,
            }
        )

    return samples


def load_reference_list(path: str) -> List[str]:
    """
    读取标准答案 JSON，支持几种形式：
      1) [{"user_input": "...", "reference": "..."}, ...]
      2) [{"reference": "..."}, ...]
      3) ["标准答案1", "标准答案2", ...]
    返回：按顺序的 reference 字符串列表
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"{path} 顶层必须是 list。")

    refs: List[str] = []
    for idx, item in enumerate(data):
        if isinstance(item, dict):
            if "reference" in item:
                refs.append(str(item["reference"]))
            elif "answer" in item:
                refs.append(str(item["answer"]))
            else:
                raise ValueError(
                    f"{path} 中第 {idx} 条记录缺少 'reference' 或 'answer' 字段。"
                )
        elif isinstance(item, str):
            refs.append(item)
        else:
            raise ValueError(
                f"{path} 中第 {idx} 条记录既不是 dict 也不是 str，无法解析。"
            )

    return refs


def build_ragas_samples(
    manual_samples: List[Dict[str, Any]],
    references: List[str],
) -> List[Dict[str, Any]]:
    """
    按顺序把「手工 RAG 结果」和「标准答案」对齐，生成 ragas 用的 4 字段结构：
      - user_input
      - response
      - retrieved_contexts
      - reference
    """
    n_manual = len(manual_samples)
    n_ref = len(references)
    n = min(n_manual, n_ref)

    if n_manual != n_ref:
        print(
            f"[WARN] 手工样本 {n_manual} 条，标准答案 {n_ref} 条，只对齐前 {n} 条。"
        )

    ragas_samples: List[Dict[str, Any]] = []
    for i in range(n):
        ms = manual_samples[i]
        ref = references[i]

        ragas_samples.append(
            {
                "user_input": ms["user_input"],
                "response": ms["response"],
                "retrieved_contexts": ms.get("retrieved_contexts", []),
                "reference": ref,
            }
        )

    return ragas_samples


def save_json(data: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    print("=== Merge manual RAG results + reference -> RAGAS JSON ===\n")

    # 1. 解析手工整理的文本
    try:
        manual_samples = parse_manual_text(MANUAL_TEXT_PATH)
    except Exception as e:
        print(f"[ERROR] 解析手工文本 {MANUAL_TEXT_PATH} 失败: {e}")
        return

    print(
        f"[INFO] 从 {MANUAL_TEXT_PATH} 解析出 {len(manual_samples)} 条样本 "
        "(user_input / response / retrieved_contexts)。"
    )

    # 2. 读取标准答案
    try:
        reference_list = load_reference_list(REFERENCE_JSON_PATH)
    except Exception as e:
        print(f"[ERROR] 读取标准答案 {REFERENCE_JSON_PATH} 失败: {e}")
        return

    print(f"[INFO] 从 {REFERENCE_JSON_PATH} 读取到 {len(reference_list)} 条标准答案。")

    # 3. 合并成 ragas 需要的四字段结构
    ragas_samples = build_ragas_samples(manual_samples, reference_list)

    # 4. 保存为 JSON
    save_json(ragas_samples, RAGAS_OUTPUT_PATH)
    print(
        f"[OK] 已生成 RAGAS 输入文件: {RAGAS_OUTPUT_PATH}，共 {len(ragas_samples)} 条样本。"
    )
    print(
        "    每条样本包含字段: user_input / response / retrieved_contexts / reference\n"
        "    可以直接在 ragas.evaluate 里用 column_map 映射：\n"
        "      question -> user_input\n"
        "      answer   -> response\n"
        "      contexts -> retrieved_contexts\n"
        "      ground_truth -> reference"
    )


if __name__ == "__main__":
    main()
