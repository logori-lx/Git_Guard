from __future__ import annotations

"""
answer_generation.py

Medical RAG - 答案生成模块

功能:
    - 读取两个 JSON 文件:
        rag_generate_and_ragas_test) rewritten_query.json:   保存改写后的用户问题
        2) top5_most_similar_vectors.json:  保存每个问题召回的 Top5 文档
    - 调用 GLM (智谱大模型) 根据 rewritten_query + Top5 文档 生成答案
    - 输出一个 JSON 文件 generated_answers.json:
        [
          {
            "user_input": "... 改写后的问题 ...",
            "response":   "... LLM 生成的回答 ...",
            "retrieved_contexts": [
                "{... 一条文档的 JSON 字符串 ...}",
                ...
            ]
          },
          ...
        ]

依赖:
    pip install "zhipuai>=2.0.0"

说明:
    - JSON 格式和项目文档中的示例完全兼容
    - 生成的 generated_answers.json 可以直接被评估脚本 ragas_evaluation.py 使用
"""

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

# ====== rag_generate_and_ragas_test. 配置区域：可以按需修改 ===============================

# 默认输入 / 输出文件名（也可以在命令行参数中覆盖）
DEFAULT_REWRITTEN_PATH = "rewritten_query.json"
DEFAULT_TOPK_PATH = "top5_most_similar_vectors.json"
DEFAULT_OUTPUT_PATH = "generated_answers.json"

# 你的智谱 API Key（也可以通过环境变量 ZHIPUAI_API_KEY 提供）
ZHIPU_API_KEY = os.getenv("ZHIPUAI_API_KEY") or "d0b8bc52cf6b4c368982dfdd32384757.UcWBjZr72H7AWgyN"

# ===============================================================

try:
    from zhipuai import ZhipuAI
except ImportError:  # pragma: no cover
    ZhipuAI = None  # type: ignore


# ---------------------------------------------------------------------
# 数据结构：单条召回文档
# ---------------------------------------------------------------------

@dataclass
class RetrievedDoc:
    """单条召回文档 / 向量条目."""

    id: int
    metadata: Dict[str, Any]
    text: str

    @classmethod
    def from_any(cls, obj: Union[str, Dict[str, Any]]) -> "RetrievedDoc":
        """
        兼容 JSON 字符串 / dict / 纯文本.

        - 如果是 JSON 字符串，会尝试解析并提取 id / metadata / text
        - 如果不是 JSON，则当成纯文本
        """
        if isinstance(obj, str):
            try:
                data = json.loads(obj)
            except json.JSONDecodeError:
                return cls(
                    id=-1,
                    metadata={"raw": obj},
                    text=obj,
                )
        else:
            data = obj

        return cls(
            id=int(data.get("id", -1)),
            metadata=data.get("metadata", {}) or {},
            text=str(data.get("text", "")),
        )


# ---------------------------------------------------------------------
# 智谱 LLM 客户端封装
# ---------------------------------------------------------------------

class ZhipuLLMClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "glm-4.5-flash",
    ) -> None:
        if ZhipuAI is None:
            raise ImportError(
                "zhipuai SDK 未安装，请先执行: pip install zhipuai"
            )

        if api_key is None:
            api_key = ZHIPU_API_KEY

        if not api_key or "在这里填你的智谱APIKey" in api_key:
            raise ValueError(
                "未提供智谱 api_key。请在文件顶部 ZHIPU_API_KEY 位置填入你的智谱 API Key，"
                "或者通过环境变量 ZHIPUAI_API_KEY 提供。"
            )

        self._client = ZhipuAI(api_key=api_key)
        self._model = model

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        # 官方 SDK 返回值结构中，choices[0].message.content 为字符串
        return str(response.choices[0].message.content).strip()  # type: ignore[attr-defined]


# ---------------------------------------------------------------------
# Prompt 构造 + 答案生成
# ---------------------------------------------------------------------

def _format_contexts_for_prompt(
    retrieved_contexts: Sequence[Union[RetrievedDoc, Dict[str, Any], str]],
    max_docs: int = 5,
) -> str:
    """
    把 TopK 召回结果格式化成 prompt 的参考资料文本.

    支持:
        - RetrievedDoc 对象
        - dict
        - JSON 字符串 / 普通字符串
    """
    lines: List[str] = []
    for idx, raw in enumerate(retrieved_contexts[:max_docs], start=1):
        if isinstance(raw, RetrievedDoc):
            doc = raw
        else:
            doc = RetrievedDoc.from_any(raw)

        meta = doc.metadata or {}
        department = meta.get("department", "")
        related = meta.get("related_disease", [])
        if isinstance(related, (list, tuple)):
            related_str = "、".join(map(str, related))
        else:
            related_str = str(related) if related else ""
        title = meta.get("title", "")
        orig_query = meta.get("query", "")

        block = [
            f"【文档 {idx}】",
            f"科室：{department}" if department else "",
            f"相关疾病：{related_str}" if related_str else "",
            f"标题：{title}" if title else "",
            f"原始问诊：{orig_query}" if orig_query else "",
            "医生回答：",
            doc.text,
        ]
        block = [line for line in block if line]
        lines.append("\n".join(block))

    return "\n\n".join(lines)


def build_answer_prompt(
    rewritten_query: str,
    retrieved_contexts: Sequence[Union[RetrievedDoc, Dict[str, Any], str]],
) -> Dict[str, str]:
    """
    根据改写后的 query 和 TopK 文档构造 LLM 的 system + user prompt.
    """
    context_text = _format_contexts_for_prompt(retrieved_contexts, max_docs=5)

    system_prompt = (
        "你是一个谨慎的中文全科医生助理，只能根据给定的【参考资料】回答问题。\n"
        "要求：\n"
        "rag_generate_and_ragas_test. 严格基于参考资料作答，避免编造资料中没有的医学结论；\n"
        "2. 不给出具体药品剂量、疗程等处方级用药建议；\n"
        "3. 如资料不足以得出明确结论，要说明不确定性并建议线下就医或咨询专业医生；\n"
        "4. 语言通俗、礼貌，适合普通患者阅读；\n"
        "5. 在回答末尾补充温馨提示，例如：如症状加重或出现严重不适，请及时就医。"
    )

    user_prompt = (
        "用户改写后的提问如下：\n"
        f"{rewritten_query.strip()}\n\n"
        "以下是与你回答相关的若干条【参考资料】，它们来自真实或模拟的问诊记录：\n"
        f"{context_text}\n\n"
        "请只根据以上参考资料，用清晰、分点的方式用中文回答用户的问题。\n"
        "如果不同资料之间存在差异，请先指出差异，再给出综合性的建议。"
    )

    return {"system_prompt": system_prompt, "user_prompt": user_prompt}


def generate_answer(
    rewritten_query: str,
    retrieved_contexts: Sequence[Union[RetrievedDoc, Dict[str, Any], str]],
    llm_client: Optional[ZhipuLLMClient] = None,
) -> str:
    """
    答案生成主函数.
    """
    if llm_client is None:
        llm_client = ZhipuLLMClient()

    prompts = build_answer_prompt(rewritten_query, retrieved_contexts)
    response = llm_client.chat(
        system_prompt=prompts["system_prompt"],
        user_prompt=prompts["user_prompt"],
    )
    return response


# ---------------------------------------------------------------------
# I/O 工具：读取 rewritten_query.json 和 top5_most_similar_vectors.json
# ---------------------------------------------------------------------

def load_rewritten_queries(path: str) -> Dict[int, str]:
    """
    读取 rewritten_query.json，返回 {id: rewritten_query} 映射.

    支持的 JSON 结构示例（任选其一）:

    rag_generate_and_ragas_test) 推荐写法（显式 id）:
        [
          { "id": rag_generate_and_ragas_test, "rewritten_query": "..." },
          { "id": 2, "rewritten_query": "..." }
        ]

    2) 简化写法（省略 id，按顺序自动编号从 rag_generate_and_ragas_test 开始）:
        [
          "问题1的改写...",
          "问题2的改写..."
        ]
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    mapping: Dict[int, str] = {}

    if isinstance(data, list):
        for idx, item in enumerate(data, start=1):
            if isinstance(item, str):
                mapping[idx] = item.strip()
            elif isinstance(item, dict):
                qid = int(item.get("id", idx))
                rq = str(item.get("rewritten_query", item.get("query", ""))).strip()
                if not rq:
                    raise ValueError(f"rewritten_query.json 中第 {idx} 条数据没有找到 rewritten_query 字段")
                mapping[qid] = rq
            else:
                raise TypeError(f"rewritten_query.json 中第 {idx} 条元素类型不支持: {type(item)}")
    elif isinstance(data, dict):
        # 兼容单个对象
        qid = int(data.get("id", 1))
        rq = str(data.get("rewritten_query", data.get("query", ""))).strip()
        if not rq:
            raise ValueError("rewritten_query.json 中没有找到 rewritten_query 字段")
        mapping[qid] = rq
    else:
        raise TypeError(f"rewritten_query.json 顶层类型必须是 list 或 dict，而不是 {type(data)}")

    return mapping


def load_topk_vectors(path: str) -> Dict[int, List[Dict[str, Any]]]:
    """
    读取 top5_most_similar_vectors.json，返回 {query_id: [doc, ...]} 映射.

    推荐 JSON 结构:

        [
          {
            "query_id": rag_generate_and_ragas_test,
            "topk_docs": [
              { "id": 10, "metadata": {...}, "text": "..." },
              ...
            ]
          },
          ...
        ]

    也兼容:
        - 把字段名写成 "id" 代替 "query_id"
        - 顶层只有一个对象而不是列表
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    result: Dict[int, List[Dict[str, Any]]] = {}

    def _add_entry(entry: Dict[str, Any], default_id: int) -> None:
        if not isinstance(entry, dict):
            raise TypeError("top5_most_similar_vectors.json 中元素必须是对象(dict)")
        qid = int(entry.get("query_id", entry.get("id", default_id)))
        docs_raw = entry.get("topk_docs") or entry.get("docs") or entry.get("vectors")
        if not isinstance(docs_raw, list):
            raise ValueError(f"query_id={qid} 的 topk_docs/docs/vectors 字段必须是列表")
        docs: List[Dict[str, Any]] = []
        for d in docs_raw:
            if isinstance(d, dict):
                docs.append(d)
            else:
                # 如果是 JSON 字符串，尝试解析一下
                if isinstance(d, str):
                    try:
                        maybe = json.loads(d)
                    except json.JSONDecodeError:
                        docs.append({"text": d})
                    else:
                        if isinstance(maybe, dict):
                            docs.append(maybe)
                        else:
                            docs.append({"text": d})
                else:
                    docs.append({"text": str(d)})
        result[qid] = docs

    if isinstance(data, list):
        for idx, item in enumerate(data, start=1):
            _add_entry(item, default_id=idx)
    elif isinstance(data, dict):
        _add_entry(data, default_id=1)
    else:
        raise TypeError(f"top5_most_similar_vectors.json 顶层类型必须是 list 或 dict，而不是 {type(data)}")

    return result


# ---------------------------------------------------------------------
# 主流程：读入两个 JSON -> 调 GLM 生成回答 -> 输出 generated_answers.json
# ---------------------------------------------------------------------

def run_answer_generation(
    rewritten_path: str = DEFAULT_REWRITTEN_PATH,
    topk_path: str = DEFAULT_TOPK_PATH,
    output_path: str = DEFAULT_OUTPUT_PATH,
) -> None:
    # rag_generate_and_ragas_test. 读取输入
    rewritten_map = load_rewritten_queries(rewritten_path)
    topk_map = load_topk_vectors(topk_path)

    if not topk_map:
        raise ValueError("top5_most_similar_vectors.json 为空，没有任何待回答的问题。")

    # 2. 初始化 LLM
    llm_client = ZhipuLLMClient()

    results: List[Dict[str, Any]] = []

    # 按 query_id 排序，保证输出顺序稳定
    for qid in sorted(topk_map.keys()):
        if qid not in rewritten_map:
            raise KeyError(
                f"在 rewritten_query.json 中找不到 id={qid} 对应的 rewritten_query，"
                f"请确保两个 JSON 使用相同的 id / query_id。"
            )

        rewritten_query = rewritten_map[qid]
        docs = topk_map[qid]

        print(f"[AnswerGen] 处理 query_id={qid} ...")

        answer = generate_answer(
            rewritten_query=rewritten_query,
            retrieved_contexts=docs,
            llm_client=llm_client,
        )

        # 根据项目文档要求，把 retrieved_contexts 存成 JSON 字符串数组
        ctx_strings = [json.dumps(doc, ensure_ascii=False) for doc in docs]

        results.append(
            {
                "user_input": rewritten_query,
                "response": answer,
                "retrieved_contexts": ctx_strings,
                "query_id": qid,  # 额外留一个字段方便调试
            }
        )

    # 3. 写出输出 JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n[AnswerGen] 共生成 {len(results)} 条回答，已保存到: {output_path}")


# ---------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Medical RAG - 答案生成模块 (rewritten_query + Top5 vectors -> generated_answers.json)"
    )
    parser.add_argument(
        "--rewritten_path",
        type=str,
        default=DEFAULT_REWRITTEN_PATH,
        help=f"改写后问题的 JSON 路径 (默认: {DEFAULT_REWRITTEN_PATH})",
    )
    parser.add_argument(
        "--topk_path",
        type=str,
        default=DEFAULT_TOPK_PATH,
        help=f"Top5 most similar vectors 的 JSON 路径 (默认: {DEFAULT_TOPK_PATH})",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help=f"生成答案输出 JSON 路径 (默认: {DEFAULT_OUTPUT_PATH})",
    )

    args = parser.parse_args()

    run_answer_generation(
        rewritten_path=args.rewritten_path,
        topk_path=args.topk_path,
        output_path=args.output_path,
    )
