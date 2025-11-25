from __future__ import annotations

"""
baseline_answer_generation.py

功能:
  - 使用“普通大模型”（这里用智谱 GLM-4.5-flash）仅基于问题本身生成答案
  - 不使用任何检索到的文档作为上下文
  - 输出 JSON: generated_plain_answers.json
      [
        {
          "id": rag_generate_and_ragas_test,
          "user_input": "...",
          "response": "...",
          "retrieved_contexts": []   # 注意: baseline 不用检索，这里是空列表
        },
        ...
      ]

依赖:
    pip install "zhipuai>=2.0.0"

环境变量:
    推荐设置环境变量 ZHIPUAI_API_KEY 为你的智谱 API key
    或者直接把 key 写到下面 ZHIPU_API_KEY 里
"""

import json
import os
from typing import Any, Dict, List

try:
    from zhipuai import ZhipuAI
except ImportError:
    ZhipuAI = None  # type: ignore


# ====== rag_generate_and_ragas_test. 配置区域 ============================================

REWRITTEN_QUERY_PATH = "rewritten_query.json"
OUTPUT_PATH = "generated_plain_answers.json"

# 智谱 API Key：优先用环境变量，其次你可以直接写死到字符串里
ZHIPU_API_KEY = os.getenv("ZHIPUAI_API_KEY") or "d0b8bc52cf6b4c368982dfdd32384757.UcWBjZr72H7AWgyN"
ZHIPU_MODEL = "glm-4.5-flash"

# =============================================================


class ZhipuLLMClient:
    """简单封装一下 GLM 聊天接口"""

    def __init__(self, api_key: str, model: str = ZHIPU_MODEL) -> None:
        if ZhipuAI is None:
            raise ImportError("未安装 zhipuai，请先运行: pip install zhipuai")

        if (not api_key) or "在这里填你的智谱APIKey" in api_key:
            raise ValueError(
                "未提供智谱 API Key。请设置环境变量 ZHIPUAI_API_KEY，"
                "或者在本文件顶部 ZHIPU_API_KEY 写入你的 key。"
            )

        self._client = ZhipuAI(api_key=api_key)
        self._model = model

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=1024,
        )
        return resp.choices[0].message.content.strip()  # type: ignore[attr-defined]


def load_rewritten_queries(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"{path} 顶层 JSON 必须是列表(list)。")

    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"{path} 中第 {idx} 条记录不是对象(dict)。")
        if "id" not in item or "rewritten_query" not in item:
            raise ValueError(
                f"{path} 中第 {idx} 条记录缺少 id 或 rewritten_query 字段。"
            )

    return data


def build_plain_system_prompt() -> str:
    """普通大模型（非 RAG）使用的系统提示词。"""
    return (
        "你是一个谨慎的中文全科医生助理。\n"
        "现在不会给你任何检索到的问诊记录或文献，只能依靠你的一般医学知识做出解释和建议。\n"
        "要求：\n"
        "rag_generate_and_ragas_test. 给出的是健康教育性质的建议，而不是正式诊断或处方；\n"
        "2. 避免给出具体药品剂量、疗程等处方级用药方案；\n"
        "3. 如存在不确定性，要明确说明，并建议在有需要时尽快线下就医或咨询专科医生；\n"
        "4. 使用通俗、礼貌、结构清晰的中文回答，适合普通患者阅读；\n"
        "5. 在回答结尾加一句温馨提示，如：如症状持续或加重，请及时就医。"
    )


def build_user_prompt(question: str) -> str:
    """把用户问题简单包装一下发给大模型。"""
    return (
        "患者的提问如下：\n"
        f"{question.strip()}\n\n"
        "请根据你的医学常识，结合临床常见情况，用清晰的分点形式回答，"
        "说明可能的原因、建议的生活方式调整，以及在什么情况下需要尽快就医或进一步检查。"
    )


def main() -> None:
    print("=== Baseline: 普通大模型答案生成 ===\n")

    # rag_generate_and_ragas_test. 读取 rewritten_query.json
    try:
        queries = load_rewritten_queries(REWRITTEN_QUERY_PATH)
    except Exception as e:
        print(f"[BASELINE] 读取 {REWRITTEN_QUERY_PATH} 失败：{e}")
        return

    print(f"[BASELINE] 成功读取 {len(queries)} 条 rewritten_query。")

    # 2. 初始化 GLM 客户端
    try:
        llm = ZhipuLLMClient(api_key=ZHIPU_API_KEY)
    except Exception as e:
        print("[BASELINE] 初始化 ZHIPU LLM 失败：", e)
        return

    system_prompt = build_plain_system_prompt()

    results: List[Dict[str, Any]] = []

    # 3. 逐条生成回答
    for item in queries:
        qid = item["id"]
        question = item["rewritten_query"]

        print(f"\n[BASELINE] 生成第 {qid} 条答案中...")
        user_prompt = build_user_prompt(question)

        try:
            answer = llm.chat(system_prompt=system_prompt, user_prompt=user_prompt)
        except Exception as e:
            print(f"[BASELINE] 调用 LLM 失败（id={qid}）：{e}")
            answer = f"[ERROR] 模型调用失败：{e}"

        results.append(
            {
                "id": qid,
                "user_input": question,
                "response": answer,
                "retrieved_contexts": [],  # baseline 不用检索
            }
        )

    # 4. 保存为 generated_plain_answers.json
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n[BASELINE] 已将普通大模型回答保存到: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
