from __future__ import annotations

"""
ragas_evaluation_from_manual_json.py

用于评估：merge_numbered_manual_to_ragas_json.py 生成的
ragas_eval_from_manual.json

该 JSON 中每条样本格式应为：
{
  "user_input": "...",
  "response": "...",
  "retrieved_contexts": ["{...}", "{...}", ...],
  "reference": "..."
}

整体流程:
    1. 读取 ragas_eval_from_manual.json
    2. 直接作为 ragas 输入（列映射为 question/answer/contexts/ground_truth）
    3. 调用 ragas + DeepSeek 进行评估，输出各项指标

依赖:
    pip install "ragas>=0.3.0" "datasets>=2.0.0" \
                "langchain>=0.2.0" "langchain-community>=0.2.0"
"""

import json
import os
from typing import Any, Dict, List

# ====== 1. 配置区域：可以按需修改 ===============================

# 输入：由 merge_numbered_manual_to_ragas_json.py 生成
RAGAS_INPUT_PATH = "ragas_eval_from_manual.json"

# 输出：评估结果
RAGAS_REPORT_PATH = "ragas_eval_report_from_manual.json"

# DeepSeek API Key（也可以通过环境变量 DEEPSEEK_API_KEY 提供）
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY") or "sk-3a59029618234b1496de29504891bf78"

# ===============================================================

# ragas + langchain 相关依赖
try:
    from ragas import evaluate  # type: ignore
    from ragas.llms import LangchainLLMWrapper  # type: ignore
    from ragas.metrics import (  # type: ignore
        LLMContextRecall,
        Faithfulness,
        FactualCorrectness,
    )
    from datasets import Dataset  # type: ignore
except ImportError:  # pragma: no cover
    evaluate = None  # type: ignore
    LangchainLLMWrapper = None  # type: ignore
    LLMContextRecall = None  # type: ignore
    Faithfulness = None  # type: ignore
    FactualCorrectness = None  # type: ignore
    Dataset = None  # type: ignore

try:
    from langchain.chat_models import init_chat_model  # type: ignore
except ImportError:  # pragma: no cover
    init_chat_model = None  # type: ignore


# ---------------------------------------------------------------------
# 2. I/O 工具函数
# ---------------------------------------------------------------------

def load_ragas_input(path: str) -> List[Dict[str, Any]]:
    """
    读取 ragas_eval_from_manual.json，校验结构是否满足 RAGAS 所需字段：
      - user_input
      - response
      - retrieved_contexts
      - reference
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到输入文件: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"{path} 顶层 JSON 必须是列表(list)。")

    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"{path} 中第 {idx} 条记录不是对象(dict)。")
        for key in ("user_input", "response", "retrieved_contexts", "reference"):
            if key not in item:
                raise ValueError(f"{path} 中第 {idx} 条记录缺少字段: {key}")

        # 防御性：保证 retrieved_contexts 为 list[str]
        if not isinstance(item["retrieved_contexts"], list):
            item["retrieved_contexts"] = [str(item["retrieved_contexts"])]
        else:
            item["retrieved_contexts"] = [
                str(c) for c in item["retrieved_contexts"]
            ]

        item["user_input"] = str(item["user_input"])
        item["response"] = str(item["response"])
        item["reference"] = str(item["reference"])

    return data


def save_json(data: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------
# 3. 调用 ragas + DeepSeek 进行评估
# ---------------------------------------------------------------------

def run_ragas(
    ragas_samples: List[Dict[str, Any]],
    deepseek_api_key: str,
) -> Dict[str, float]:
    """
    使用 DeepSeek 作为判分 LLM，对 ragas_samples 进行评估。

    ragas_samples 中每条样本字段：
      - user_input       -> question
      - response         -> answer
      - retrieved_contexts -> contexts
      - reference        -> ground_truth
    """
    if evaluate is None or LangchainLLMWrapper is None or Dataset is None:
        raise ImportError(
            "ragas 或 datasets 未正确安装。请先执行：\n"
            "  pip install 'ragas>=0.3.0' 'datasets>=2.0.0'"
        )
    if LLMContextRecall is None or Faithfulness is None or FactualCorrectness is None:
        raise ImportError(
            "ragas.metrics 未正确导入，请检查 ragas 版本。"
        )
    if init_chat_model is None:
        raise ImportError(
            "langchain 未安装，请先执行：\n"
            "  pip install 'langchain>=0.2.0' 'langchain-community>=0.2.0'"
        )

    if not deepseek_api_key or "在这里填你的DeepSeek_APIKey" in deepseek_api_key:
        raise ValueError(
            "未提供 DeepSeek API Key。请在文件顶部 DEEPSEEK_API_KEY 位置填入你的 Key，"
            "或者通过环境变量 DEEPSEEK_API_KEY 提供。"
        )

    if not ragas_samples:
        raise ValueError("ragas_samples 为空，无法进行评估。")

    # 1. 构造 HuggingFace Dataset
    dataset = Dataset.from_list(ragas_samples)  # type: ignore[arg-type]

    # 2. 初始化 DeepSeek 作为评估 LLM
    llm = init_chat_model(
        model="deepseek-chat",
        api_key=deepseek_api_key,
        api_base="https://api.deepseek.com/",
        temperature=0,
        model_provider="deepseek",
    )
    evaluator_llm = LangchainLLMWrapper(llm)

    # 3. 调用 ragas.evaluate
    result = evaluate(
        dataset=dataset,
        metrics=[
            LLMContextRecall(),
            Faithfulness(),
            FactualCorrectness(),
        ],
        llm=evaluator_llm,
        column_map={
            "question": "user_input",
            "answer": "response",
            "contexts": "retrieved_contexts",
            "ground_truth": "reference",
        },
    )

    # 4. 将结果转成简单的 {metric: score} 字典
    metrics_dict: Dict[str, float] = {}

    try:
        if hasattr(result, "to_pandas"):
            df = result.to_pandas()  # type: ignore[attr-defined]
            for col in df.columns:
                try:
                    metrics_dict[col] = float(df[col].mean())
                except Exception:
                    continue
        else:
            metrics_dict = dict(result)  # type: ignore[arg-type]
    except Exception:
        try:
            metrics_dict = dict(result)  # type: ignore[arg-type]
        except Exception:
            metrics_dict = {}

    return metrics_dict


# ---------------------------------------------------------------------
# 4. 主入口
# ---------------------------------------------------------------------

def main() -> None:
    print("=== Medical RAG: RAGAS 评估（来自 ragas_eval_from_manual.json）===\n")

    # 1. 读取 ragas_eval_from_manual.json
    try:
        ragas_samples = load_ragas_input(RAGAS_INPUT_PATH)
    except Exception as e:
        print(f"[RAGAS] 读取 {RAGAS_INPUT_PATH} 失败：{e}")
        return

    print(f"[RAGAS] 成功读取 {RAGAS_INPUT_PATH}，共 {len(ragas_samples)} 条样本。")

    # 2. 调用 ragas + DeepSeek 评估
    try:
        metrics = run_ragas(ragas_samples, deepseek_api_key=DEEPSEEK_API_KEY)
    except ImportError as e:
        print("[RAGAS] 依赖未安装，无法执行评估。")
        print("错误信息：", e)
        return
    except Exception as e:
        print("[RAGAS] 调用 ragas 评估时出错：", e)
        return

    # 3. 打印并保存评估结果
    if metrics:
        print("\n[RAGAS] 评估结果（各指标平均分）:")
        for name, score in metrics.items():
            try:
                print(f"  {name}: {float(score):.4f}")
            except Exception:
                print(f"  {name}: {score}")
        save_json(metrics, RAGAS_REPORT_PATH)
        print(f"\n[RAGAS] 评估结果已写入: {RAGAS_REPORT_PATH}")
    else:
        print("[RAGAS] 评估返回结果为空，请检查 ragas 版本或输入数据。")


if __name__ == "__main__":
    main()
