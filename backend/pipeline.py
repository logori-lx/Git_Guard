import os
from typing import List, Dict, Tuple, Optional
from query_constructing.query_constructor import QueryConstructor
from retrieve.retrieval import Retrieval, RetrievalMethod
from retrieve.rerank import Reranker


_retriever: Optional[Retrieval] = None
_query_constructor: Optional[QueryConstructor] = None


def _get_query_constructor() ->QueryConstructor:
    global _query_constructor
    if _query_constructor is None:
        api_key = os.getenv("MEDICAL_RAG")
        _query_constructor = QueryConstructor(api_key=api_key)
    return _query_constructor

def _get_retriever() -> Retrieval:
    global _retriever
    if _retriever is None:
        # 若本地向量库未初始化，ret.py 内部会抛出清晰的报错提示
        _query_constructor = _get_query_constructor()
        _retriever = Retrieval(_query_constructor)
    return _retriever


def initialize_rag_system():
    """
    预初始化 RAG 系统（包括 retriever 和 client）。
    建议在应用启动时调用，避免首次请求时的初始化延迟。
    """
    print("正在初始化 RAG 系统...")
    try:
        retriever = _get_retriever()
        print("✓ Retriever 初始化完成（包括 BM25 索引）")
    except Exception as e:
        print(f"✗ Retriever 初始化失败: {e}")
    
    print("RAG 系统初始化完成！")
    return retriever



def retrieve_and_generate(
    question: str,
    search_type: str=RetrievalMethod.HYBRID.value,
    model: str = "glm-4",
    temperature: float = 1.0,
    max_tokens: int = 65536,
) -> Tuple[str, List[Dict]]:
    """
    完整 RAG：检索 -> 提示构建 -> 大模型生成。

    返回：
    - answer: 最终回答（中文）
    - context: 引用的检索结果（便于前端展示溯源）
    """
    retriever = _get_retriever()
    rewritten_query = _get_query_constructor().get_query(question)
    # 召回
    contexts = retriever.retrieve(
        query=rewritten_query, top_k=50,  retrieval_type=search_type
    )
    reranker = Reranker()
    reranked_contexts = reranker.rerank(rewritten_query,contexts, top_k=5)
    answer = _get_query_constructor().process_medical_query(question=rewritten_query,
                                                            context=reranked_contexts,
                                                            model=model, 
                                                            temperature=temperature, 
                                                            max_tokens=max_tokens)
    final_context = []
    for context in reranked_contexts:
        final_context.append(
            {
                "ask": context["ask"],
                "answer": context["answer"],
                "department": context["department"],
            }
        )
    return answer, final_context


__all__ = ["retrieve_and_generate", "initialize_rag_system"]
