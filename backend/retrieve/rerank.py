import os
import requests
import sys
# 获取项目根目录的绝对路径（当前文件在 retrieve 文件夹下，上一级就是根目录）
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# 将根目录加入 Python 系统路径
sys.path.append(root_path)
from retrieve.retrieval import Retrieval, RetrievalMethod
from query_constructing.query_constructor import QueryConstructor


API_KEY = os.getenv("MEDICAL_RAG")

class Reranker:
    def __init__(self, model="rerank"):
        self.model = model
        self.url = "https://open.bigmodel.cn/api/paas/v4/rerank"
    def rerank(self,query, documents, top_k = 5):
        document_texts = [doc["answer"] for doc in documents]
        payload = {
            "model": self.model,
            "query": query,
            "top_n": top_k,
            "documents": document_texts
        }
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        try:
            response = requests.post(self.url, json=payload, headers=headers)
            result_data = response.json()
            if 'results' in result_data:
                reranked_results = []
                for item in result_data["results"]:
                    original_index = item["index"]
                    original_doc = documents[original_index]

                    reranked_doc = original_doc.copy()
                    reranked_doc["rerank_score"] = item.get("relevance_score", 0)
                    reranked_doc["rerank_index"] = item["index"]
                    reranked_results.append(reranked_doc)

                print(f"重排序完成，返回前{len(reranked_results)}个结果")
                return reranked_results
            else:
                print("重排序API返回格式异常，使用原始排序")
                return documents[:top_k]
        except Exception as e:
            print(f"重排序失败: {e}，使用原始排序")
            return documents[:top_k]
    
    
if __name__ == "__main__":
    query = "得了糖尿病需要注意什么"
    query_constructor = QueryConstructor(API_KEY)
    retriever = Retrieval(query_constructor=query_constructor)
    documents = retriever.retrieve(retrieval_type=RetrievalMethod.BM25.value,query=query, top_k=50) 
    rerank = Reranker()
    results = rerank.rerank(query, documents)
    print(results)



