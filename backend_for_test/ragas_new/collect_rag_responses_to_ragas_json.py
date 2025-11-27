# -*- coding: utf-8 -*-
"""
collect_rag_responses_to_ragas_json.py

功能：
  1. 从 rewritten_query.json 读取你设计好的测试问题
  2. 把每个问题发给正在运行的 RAG 系统（http://localhost:5173/）
  3. 从返回结果里提取：
        - 回答（response）
        - 检索到的上下文（retrieved_contexts）
     然后和标准答案（reference_answers.json）对齐，
     写成一个可以直接给 ragas 用的 JSON：
        [
          {
            "id": 1,
            "user_input": "...",
            "response": "...",
            "retrieved_contexts": ["...", "..."],
            "reference": "..."
          },
          ...
        ]

使用前准备：
  1. 安装依赖： pip install requests
  2. 和本脚本同目录下放好：
       - rewritten_query.json
       - reference_answers.json
  3. 根据你自己 RAG 后端的接口，把 BACKEND_API_URL 和 build_request_payload() 改对。
"""

import json
import time
import re
from typing import Any, Dict, List, Tuple
import requests

# ===================== 需要你根据自己系统改的配置 =====================

# 你的 RAG 系统后端 API 地址
# 如果前端 Vite 把 /api/* 代理到 FastAPI，很多项目是这样：
#   例如： http://localhost:5173/api/chat  或  /api/qa
# 你只要改成你实际在 Network 面板看到的那个接口就行。
BACKEND_API_URL = "http://localhost:5173/api/chat"  # TODO: 修改为你项目真实的接口路径

# 请求之间的间隔，避免一瞬间打太多请求（按需调节）
SLEEP_SECONDS = 1.0

# 输入 / 输出文件名
REWRITTEN_QUERY_PATH = "rewritten_query.json"      # 你那 20 个改写后问题
REFERENCE_ANSWERS_PATH = "reference_answers.json"  # 对应的标准答案
OUTPUT_JSON_PATH = "ragas_eval_input_rag.json"     # 本脚本生成，给 ragas 用


def build_request_payload(question: str) -> Dict[str, Any]:
    """
    根据你后端的接口格式构造请求体。

    TODO: 这里一定要根据你自己后端实际的参数名改！！
    比如：
        - 有的接口是 {"query": question}
        - 有的是 {"message": question}
        - 有的是 {"question": question}

    打开浏览器 F12 -> Network，看看前端发给后端的请求体，照着改就行。
    """
    return {
        "query": question   # TODO: 如果你后端字段名不是 "query"，在这里改
    }

# =====================================================================


# ---------------------------------------------------------------------
# 1. 读取 rewritten_query.json 和 reference_answers.json
#    假设格式为：
#    rewritten_query.json  : [{"id": 1, "rewritten_query": "..."} , ...]
#    reference_answers.json: [{"id": 1, "reference": "..."}        , ...]
# ---------------------------------------------------------------------

def load_rewritten_queries(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions: List[Dict[str, Any]] = []

    if isinstance(data, list):
        for idx, item in enumerate(data):
            if isinstance(item, dict):
                qid = item.get("id", idx + 1)
                qtext = (
                    item.get("rewritten_query")
                    or item.get("question")
                    or item.get("user_input")
                )
            else:
                qid = idx + 1
                qtext = str(item)

            if not qtext:
                raise ValueError(f"rewritten_query.json 第 {idx} 条没有找到问题字段。")

            questions.append({"id": qid, "question": str(qtext)})
    else:
        # 兜底：如果是 { "1": "问题1", "2": "问题2", ... } 这种
        for k, v in data.items():
            try:
                qid = int(k)
            except ValueError:
                qid = k
            questions.append({"id": qid, "question": str(v)})

    return questions


def load_reference_answers(path: str) -> Dict[Any, str]:
    """
    把 reference_answers.json 转成 {id: reference_text} 的字典方便对齐。
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ref_map: Dict[Any, str] = {}

    if isinstance(data, list):
        for idx, item in enumerate(data):
            if not isinstance(item, dict):
                continue
            qid = item.get("id", idx + 1)
            ref = item.get("reference") or item.get("answer") or ""
            ref_map[qid] = str(ref)
    elif isinstance(data, dict):
        # 例如 { "1": {"reference": "..."}, ... } 或 { "1": "..." }
        for k, v in data.items():
            try:
                qid = int(k)
            except ValueError:
                qid = k
            if isinstance(v, dict):
                ref_map[qid] = str(v.get("reference") or v.get("answer") or "")
            else:
                ref_map[qid] = str(v)
    else:
        raise ValueError("reference_answers.json 格式不支持，请改成 list 或 dict。")

    return ref_map


# ---------------------------------------------------------------------
# 2. 调用 RAG 后端并解析回答 + 检索上下文
# ---------------------------------------------------------------------

def call_rag_backend(question: str) -> Dict[str, Any]:
    """
    调用你自己的 RAG 后端。默认用 POST + JSON。

    如果你后端用的是 GET 或者路径参数，同样在这里改就行。
    """
    payload = build_request_payload(question)
    resp = requests.post(BACKEND_API_URL, json=payload, timeout=120)
    resp.raise_for_status()

    try:
        data = resp.json()
    except ValueError:
        # 返回不是 JSON，就用纯文本兜底
        data = {"answer": resp.text}

    return data


def split_answer_and_reference_cases(text: str) -> Tuple[str, List[str]]:
    """
    针对你给的示例答案做的解析：

    例子里回答的结构大概是：
        主回答...
        请注意，我的建议不能替代专业医疗意见...
        隐藏参考案例 Reference cases ▲
        参考案例（Reference cases）
        Case 1
        Question: ...
        Answer: ...
        科室：...
        Case 2
        ...

    我们希望：
      - response 只保留「主回答」部分
      - retrieved_contexts 每个 Case 当成一个 context

    若文本里没有“参考案例”相关字样，就原样返回。
    """
    if not text:
        return "", []

    markers = [
        "参考案例（Reference cases）",
        "参考案例 (Reference cases)",
        "参考案例 Reference cases",
        "参考案例",
        "Reference cases",
    ]
    idx = -1
    for m in markers:
        idx = text.find(m)
        if idx != -1:
            break

    # 没找到「参考案例」之类的标记，说明整个都是主回答
    if idx == -1:
        return text.strip(), []

    main_answer = text[:idx].strip()
    refs_text = text[idx:].strip()

    # 正则匹配 "Case 1 ... Case 2 ... Case 3 ..."
    pattern = re.compile(r"(Case\s+\d+[\s\S]*?)(?=(Case\s+\d+)|\Z)", re.MULTILINE)
    contexts: List[str] = [m.group(1).strip() for m in pattern.finditer(refs_text)]

    if not contexts:
        contexts = [refs_text]

    return main_answer, contexts


def extract_answer_and_contexts(api_data: Dict[str, Any]) -> Tuple[str, List[str]]:
    """
    从后端返回的数据中抽取：
        - answer（主回答）
        - contexts（检索到的文档/参考案例）
    """
    # 1) 先找回答字段
    answer_raw = (
        api_data.get("answer")
        or api_data.get("response")
        or api_data.get("output")
        or api_data.get("content")
    )
    if answer_raw is None:
        # 实在找不到就把整个 JSON 序列化成文本
        answer_raw = json.dumps(api_data, ensure_ascii=False)

    answer_raw = str(answer_raw)

    # 2) 尝试从结构化字段里找 context
    contexts: List[str] = []

    # 常见几种字段名，按你自己的后端可以再加
    candidate_keys = [
        "contexts",
        "retrieved_contexts",
        "reference_cases",
        "source_documents",
        "docs",
    ]
    for key in candidate_keys:
        if key in api_data and api_data[key]:
            raw_ctxs = api_data[key]
            # 可能是 dict / list / 单个对象
            if isinstance(raw_ctxs, dict):
                raw_ctxs = list(raw_ctxs.values())
            if not isinstance(raw_ctxs, list):
                raw_ctxs = [raw_ctxs]

            for c in raw_ctxs:
                if isinstance(c, str):
                    contexts.append(c)
                elif isinstance(c, dict):
                    # 常见字段 text / content / answer
                    t = (
                        c.get("text")
                        or c.get("content")
                        or c.get("answer")
                        or json.dumps(c, ensure_ascii=False)
                    )
                    contexts.append(str(t))
                else:
                    contexts.append(str(c))
            break  # 找到一个字段就够了

    # 3) 根据示例文本解析“参考案例”
    main_answer, parsed_cases = split_answer_and_reference_cases(answer_raw)

    # 如果结构化字段里没找到上下文，就用解析出来的参考案例
    if not contexts and parsed_cases:
        contexts = parsed_cases

    return main_answer, contexts


# ---------------------------------------------------------------------
# 3. 主流程：循环提问 -> 收集结果 -> 保存为 ragas JSON
# ---------------------------------------------------------------------

def main() -> None:
    print("=== Collect RAG responses into RAGAS JSON ===\n")

    # 1) 加载测试问题 & 标准答案
    questions = load_rewritten_queries(REWRITTEN_QUERY_PATH)
    ref_map = load_reference_answers(REFERENCE_ANSWERS_PATH)

    print(f"[INFO] 从 {REWRITTEN_QUERY_PATH} 读取到 {len(questions)} 个问题。")
    print(f"[INFO] 从 {REFERENCE_ANSWERS_PATH} 读取到 {len(ref_map)} 条标准答案。\n")

    samples_for_ragas: List[Dict[str, Any]] = []

    for i, q in enumerate(questions, start=1):
        qid = q["id"]
        qtext = q["question"]
        print(f"[{i}/{len(questions)}] 发送问题 (id={qid}): {qtext[:40]}...")

        try:
            api_data = call_rag_backend(qtext)
            answer, contexts = extract_answer_and_contexts(api_data)
        except Exception as e:
            print(f"  [ERROR] 调用 RAG 后端失败: {e}")
            answer = ""
            contexts = []

        ref = ref_map.get(qid, "")

        sample = {
            "id": qid,
            "user_input": qtext,
            "response": answer,
            "retrieved_contexts": contexts,
            "reference": ref,
        }
        samples_for_ragas.append(sample)

        # 打个简单预览
        print(f"  [OK] 回答长度: {len(answer)}，上下文条数: {len(contexts)}")

        # 避免过快打爆后端
        time.sleep(SLEEP_SECONDS)

    # 4) 写出 ragas 可直接使用的 JSON
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(samples_for_ragas, f, ensure_ascii=False, indent=2)

    print(f"\n[DONE] 已生成 RAGAS 输入文件: {OUTPUT_JSON_PATH}")
    print("      结构包含四个字段：user_input, response, retrieved_contexts, reference")
    print("      你可以直接在 ragas 评估脚本里用 Dataset.from_list(...) 或 EvaluationDataset.from_list(...) 读取。")


if __name__ == "__main__":
    main()
