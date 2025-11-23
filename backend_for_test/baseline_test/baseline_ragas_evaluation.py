# from __future__ import annotations
#
# """
# baseline_ragas_evaluation.py
#
# 功能:
#   - 使用 ragas + DeepSeek 对“普通大模型”（无 RAG）生成的答案做评估
#   - 使用与 RAG 系统一样的 rewritten_query 和 reference（从 ragas_evaluation.py 里复用）
#   - 方便对比:
#         RAG 系统       --> ragas_eval_report.json
#         普通大模型     --> test1_ragas_eval_report_plain.json
#
# 前置:
#     rag_generate_and_ragas_test. 已经运行 baseline_answer_generation.py
#        得到 generated_plain_answers.json
#     2. 项目中已有 ragas_evaluation.py，且其中包含:
#          - REFERENCE_LIST
#          - load_generated_answers
#          - build_ragas_input
#          - save_json
#          - run_ragas
#
# 依赖:
#     与 ragas_evaluation.py 相同:
#     pip install "ragas>=0.3.0" "datasets>=2.0.0" \
#                 "langchain>=0.2.0" "langchain-community>=0.2.0"
#
# 环境变量:
#     DEEPSEEK_API_KEY  (或在本文件中写死)
# """
#
# import os
# from typing import Any, Dict, List
#
# # 从你现有的 ragas_evaluation.py 中导入共用内容
# from ragas_evaluation import (  # type: ignore
#     REFERENCE_LIST,
#     load_generated_answers,
#     build_ragas_input,
#     save_json,
#     run_ragas,
# )
#
# # ====== 配置区域 ===============================================
#
# GENERATED_PLAIN_ANSWERS_PATH = "generated_plain_answers.json"
#
# RAGAS_INPUT_PLAIN_PATH = "ragas_eval_input_plain.json"
# RAGAS_REPORT_PLAIN_PATH = "test1_ragas_eval_report_plain.json"
#
# DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY") or "sk-3a59029618234b1496de29504891bf78"
#
# # ===============================================================
#
#
# def main() -> None:
#     print("=== Medical RAG: Baseline 普通大模型 RAGAS 评估 ===\n")
#
#     # rag_generate_and_ragas_test. 读取 generated_plain_answers.json
#     try:
#         generated_plain: List[Dict[str, Any]] = load_generated_answers(
#             GENERATED_PLAIN_ANSWERS_PATH
#         )
#     except Exception as e:
#         print(f"[BASELINE-RAGAS] 读取 {GENERATED_PLAIN_ANSWERS_PATH} 失败：{e}")
#         return
#
#     print(
#         f"[BASELINE-RAGAS] 成功读取 generated_plain_answers.json，共 {len(generated_plain)} 条样本。"
#     )
#
#     # 2. 构造 ragas 输入（对齐 reference）
#     ragas_samples = build_ragas_input(generated_plain, REFERENCE_LIST)
#     save_json(ragas_samples, RAGAS_INPUT_PLAIN_PATH)
#     print(f"[BASELINE-RAGAS] 已将评估数据写入: {RAGAS_INPUT_PLAIN_PATH}")
#
#     # 3. 调用 ragas + DeepSeek 进行评估
#     try:
#         metrics = run_ragas(ragas_samples, deepseek_api_key=DEEPSEEK_API_KEY)
#     except ImportError as e:
#         print("[BASELINE-RAGAS] 依赖未安装，无法执行评估。")
#         print("错误信息：", e)
#         return
#     except Exception as e:
#         print("[BASELINE-RAGAS] 调用 ragas 评估时出错：", e)
#         print("如果暂时只需要生成 ragas_eval_input_plain.json，可以忽略此错误。")
#         return
#
#     # 4. 打印并保存评估结果
#     if metrics:
#         print("\n[BASELINE-RAGAS] 评估结果（各指标平均分）:")
#         for name, score in metrics.items():
#             try:
#                 print(f"  {name}: {float(score):.4f}")
#             except Exception:
#                 print(f"  {name}: {score}")
#         save_json(metrics, RAGAS_REPORT_PLAIN_PATH)
#         print(f"\n[BASELINE-RAGAS] 评估结果已写入: {RAGAS_REPORT_PLAIN_PATH}")
#     else:
#         print("[BASELINE-RAGAS] 评估返回结果为空，请检查 ragas 版本或输入数据。")
#
#
# if __name__ == "__main__":
#     main()

from __future__ import annotations

"""
baseline_ragas_evaluation.py

功能:
  rag_generate_and_ragas_test. 读取 baseline_answer_generation.py 生成的 generated_plain_answers.json
  2. 读取 top5_most_similar_vectors.json (结构: query_id + topk_docs)
  3. 按 id 对齐，生成 ragas_eval_input_plain.json，结构为:
        [
          {
            "user_input": "...",
            "response": "...",
            "retrieved_contexts": ["文档1文本...", "文档2文本...", ...],
            "reference": "标准答案文本..."
          },
          ...
        ]
  4. 可选：调用 ragas_evaluation.run_ragas 跑 ragas，输出 ragas_eval_report_plain.json
"""

import json
import os
from typing import Any, Dict, List

# 从你自己的 ragas_evaluation.py 里复用标准答案和 ragas 调用函数
from ragas_evaluation import (  # type: ignore
    REFERENCE_LIST,
    run_ragas,
)

# ================== 配置区域 ========================

# baseline 生成的答案文件
GENERATED_PLAIN_ANSWERS_PATH = "generated_plain_answers.json"

# 你的 topk 文件（刚才发的那个）
TOPK_VECTORS_PATH = "top5_most_similar_vectors.json"

# 输出：带四个字段的 ragas 输入
RAGAS_INPUT_PLAIN_PATH = "ragas_eval_input_plain.json"

# 输出：baseline 的 ragas 评估报告
RAGAS_REPORT_PLAIN_PATH = "ragas_eval_report_plain.json"

# DeepSeek API Key（也可以只用环境变量 DEEPSEEK_API_KEY）
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY") or "sk-3a59029618234b1496de29504891bf78"

# ==================================================


def load_json_list(path: str) -> List[Any]:
    """读取顶层为 list 的 JSON 文件。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} 顶层 JSON 必须是 list。")
    return data


def load_generated_plain_answers(path: str) -> List[Dict[str, Any]]:
    """
    generated_plain_answers.json 结构约定:
    [
      {
        "id": rag_generate_and_ragas_test,
        "user_input": "...",
        "response": "...",
        "retrieved_contexts": []   # baseline 生成时为空，这里不依赖它
      },
      ...
    ]
    """
    data = load_json_list(path)
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"{path} 第 {idx} 条记录不是对象(dict)。")
        for k in ("id", "user_input", "response"):
            if k not in item:
                raise ValueError(f"{path} 第 {idx} 条记录缺少字段: {k}")
    return data  # type: ignore[return-value]


def load_topk_contexts(path: str) -> Dict[int, List[str]]:
    """
    精确适配你这份 top5_most_similar_vectors.json 的结构:

    [
      {
        "query_id": rag_generate_and_ragas_test,
        "topk_docs": [
          {"id": rag_generate_and_ragas_test, "metadata": {...}, "text": "文档1..."},
          {"id": 2, "metadata": {...}, "text": "文档2..."}
        ]
      },
      ...
    ]

    返回字典:
        { query_id: [ "文档1...", "文档2...", ... ] }
    """
    data = load_json_list(path)
    id2ctx: Dict[int, List[str]] = {}

    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            continue

        qid = item.get("query_id")
        if qid is None:
            # 为了安全，如果有别的字段名，也可以在这里加兼容
            # qid = item.get("id") or item.get("qid") ...
            continue

        docs = item.get("topk_docs") or []
        if not isinstance(docs, list):
            docs = [docs]

        ctx_texts: List[str] = []
        for d in docs:
            if isinstance(d, dict):
                # 优先取 text 字段
                text = d.get("text")
                if text is None:
                    text = json.dumps(d, ensure_ascii=False)
                ctx_texts.append(str(text))
            else:
                ctx_texts.append(str(d))

        id2ctx[int(qid)] = ctx_texts

    return id2ctx


def build_reference_map() -> Dict[int, str]:
    """
    把 REFERENCE_LIST 映射为 {id: reference_text}.

    约定:
      - REFERENCE_LIST[0] 对应 id = rag_generate_and_ragas_test
      - REFERENCE_LIST[rag_generate_and_ragas_test] 对应 id = 2
      - ...
    """
    ref_map: Dict[int, str] = {}
    for idx, item in enumerate(REFERENCE_LIST):
        ref_map[idx + 1] = item.get("reference", "")
    return ref_map


def build_ragas_input_plain(
    generated_plain: List[Dict[str, Any]],
    id2ctx: Dict[int, List[str]],
    ref_map: Dict[int, str],
) -> List[Dict[str, Any]]:
    """
    整合 baseline 生成结果 + TopK 上下文 + reference，
    输出 ragas_eval_input_plain.json 所需结构:

    [
      {
        "user_input": "...",
        "response": "...",
        "retrieved_contexts": ["文档1...", "文档2...", ...],
        "reference": "标准答案..."
      },
      ...
    ]
    """
    samples: List[Dict[str, Any]] = []

    for item in generated_plain:
        qid_raw = item.get("id")
        if qid_raw is None:
            continue

        try:
            qid = int(qid_raw)
        except Exception:
            continue

        user_input = item.get("user_input", "")
        response = item.get("response", "")

        ctx_texts = id2ctx.get(qid, [])
        reference = ref_map.get(qid, "")

        samples.append(
            {
                "user_input": user_input,
                "response": response,
                "retrieved_contexts": ctx_texts,
                "reference": reference,
            }
        )

    return samples


def save_json(data: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    print("=== Medical RAG: Baseline RAGAS 输入构造（四参数版，适配 topk_docs）===\n")

    # rag_generate_and_ragas_test. 读取 baseline 大模型回答
    try:
        generated_plain = load_generated_plain_answers(GENERATED_PLAIN_ANSWERS_PATH)
    except Exception as e:
        print(f"[BASELINE-RAGAS] 读取 {GENERATED_PLAIN_ANSWERS_PATH} 失败：{e}")
        return
    print(
        f"[BASELINE-RAGAS] 成功读取 generated_plain_answers.json，共 {len(generated_plain)} 条样本。"
    )

    # 2. 读取 TopK 上下文
    try:
        id2ctx = load_topk_contexts(TOPK_VECTORS_PATH)
    except Exception as e:
        print(f"[BASELINE-RAGAS] 读取 {TOPK_VECTORS_PATH} 失败：{e}")
        return
    print(
        f"[BASELINE-RAGAS] 成功读取 top5_most_similar_vectors.json，共 {len(id2ctx)} 个 query_id 有上下文。"
    )

    # 3. 构造 id -> reference 的映射
    ref_map = build_reference_map()

    # 4. 整合成 ragas_eval_input_plain.json
    ragas_samples = build_ragas_input_plain(generated_plain, id2ctx, ref_map)
    if not ragas_samples:
        print("[BASELINE-RAGAS] ragas_samples 为空，请检查 id 是否对应。")
        return

    save_json(ragas_samples, RAGAS_INPUT_PLAIN_PATH)
    print(
        f"[BASELINE-RAGAS] 已生成 {RAGAS_INPUT_PLAIN_PATH}，共 {len(ragas_samples)} 条样本。"
    )

    # 小检查：打印第一条 retrieved_contexts 数量，确认不为空
    first = ragas_samples[0]
    print(
        f"[BASELINE-RAGAS] 示例: 第 rag_generate_and_ragas_test 条样本 retrieved_contexts 数量 = "
        f"{len(first.get('retrieved_contexts', []))}"
    )

    # 5. 可选：调用 ragas 评估（不需要可以注释掉下面这段）
    try:
        metrics = run_ragas(ragas_samples, deepseek_api_key=DEEPSEEK_API_KEY)
    except Exception as e:
        print("[BASELINE-RAGAS] 调用 ragas 评估失败，可忽略此错误，仅使用生成的 JSON 即可。")
        print("错误信息：", e)
        return

    if metrics:
        print("\n[BASELINE-RAGAS] 评估结果（各指标平均分）:")
        for name, score in metrics.items():
            try:
                print(f"  {name}: {float(score):.4f}")
            except Exception:
                print(f"  {name}: {score}")
        save_json(metrics, RAGAS_REPORT_PLAIN_PATH)
        print(f"\n[BASELINE-RAGAS] 评估结果已写入: {RAGAS_REPORT_PLAIN_PATH}")
    else:
        print("[BASELINE-RAGAS] 评估结果为空，请检查 ragas 版本或输入数据。")


if __name__ == "__main__":
    main()
