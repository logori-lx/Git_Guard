import os
import sys
import requests
from typing import List, Dict
from retrieval import Retrieval, RetrievalMethod
from query_constructing.query_constructor import QueryConstructor
API_KEY = os.getenv("MEDICAL_RAG")

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_path)


class Rerank:
    def __init__(self, model="rerank"):
        self.model = model
        self.api_key = API_KEY
        self.url = "https://open.bigmodel.cn/api/paas/v4/rerank"

    def rerank(self, query: str, docs: List[Dict], top_k: int = 5) -> List[Dict]:
        if not docs:
            return []

        # 优先尝试 API 排序（如果拿到了 key）
        if self.api_key:
            try:
                return self._api_rerank(query, docs, top_k)
            except Exception as e:
                print(f"[Rerank-API失败] {e}")

        # 无 key 或失败 → 本地 score 兜底
        return self._score_rerank(docs, top_k)

    def _api_rerank(self, query: str, docs: List[Dict], top_k: int) -> List[Dict]:
        payload = {
            "model": self.model,
            "query": query,
            "documents": [d.get("answer", "") for d in docs],
            "top_n": top_k
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        resp = requests.post(self.url, json=payload, headers=headers).json()

        # API 返回异常 → 兜底排序
        if "results" not in resp:
            print("[API返回异常，将使用 score 排序]")
            return self._score_rerank(docs, top_k)

        # 按 relevance_score 排序取 topK（保持原结构不新增字段）
        ranked = sorted(resp["results"], key=lambda x: x["relevance_score"], reverse=True)
        return [docs[i["index"]] for i in ranked][:top_k]

    def _score_rerank(self, docs: List[Dict], top_k: int) -> List[Dict]:
        docs_sorted = sorted(docs, key=lambda d: d.get("score", 0), reverse=True)
        return docs_sorted[:top_k]


if __name__ == "__main__":

    query = "糖尿病的症状有哪些"
    qc = QueryConstructor(API_KEY)
    retriever = Retrieval(query_constructor=qc)

    docs = retriever.retrieve(RetrievalMethod.BM25.value, query, top_k=50)
    reranker = Rerank()
    results = reranker.rerank(query, docs, top_k=5)
