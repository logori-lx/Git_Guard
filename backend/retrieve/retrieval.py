import os
import chromadb
from typing import List, Dict
from pyparsing import Enum
from zai import ZhipuAiClient  # 确保 zai.py 在PYTHONPATH中
import pandas as pd
from llama_index.core import Document
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.node_parser import SentenceSplitter
import jieba
import sys
import sys
# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（根据实际目录结构调整）
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
# 将根目录添加到Python路径
if root_dir not in sys.path:
    sys.path.append(root_dir)
from query_constructing.query_constructor import QueryConstructor

# 获取项目根目录的绝对路径（当前文件在 retrieve 文件夹下，上一级就是根目录）
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# 将根目录加入 Python 系统路径
sys.path.append(root_path)
API_KEY = os.getenv("MEDICAL_RAG")
COLLECTION_NAME = "medical_db"
# 解决路径问题
KEYWORDS_FILE = os.path.join(root_path,"retrieve","disease_names_processed.csv")
DB_PATH = os.path.join(root_path,"DATA","chroma_db")


class RetrievalMethod(Enum):
    VECTOR = "vector"
    HYBRID = "hybrid"
    BM25 = "bm25"
class ZhipuEmbeddingFunction:
    """智谱Embedding3嵌入函数"""
    """（添加name属性）"""
    def name(self):
        """返回嵌入模型名称（Chroma要求的方法）"""
        return "zhipu-embedding-3"  # 关键修复：将name改为方法
    def __call__(self, input: List[str]) -> List[List[float]]:
        self.zhipu_api_key = API_KEY
        if not self.zhipu_api_key:
             raise ValueError("ZHIPU_API_KEY environment variable not set.")
        client = ZhipuAiClient(api_key= self.zhipu_api_key)
        response = client.embeddings.create(
                        model="embedding-3", #填写需要调用的模型编码
                        input=input,
                    )
        return [data_point.embedding for data_point in response.data]
    def embed_query(self, input) -> List[float]:
        """
        对查询字符串进行嵌入。
        
        :param query: 用户输入的查询字符串
        :return: 嵌入向量
        """
        return self.__call__(input)
    
class Retrieval:
    def __init__(self, query_constructor: QueryConstructor, collection_name= COLLECTION_NAME, embedding_function = ZhipuEmbeddingFunction()):
        self.client = chromadb.PersistentClient(path=DB_PATH)
        self.collection_name = collection_name
        self.embedding_function = embedding_function
        self.keywords = pd.read_csv(KEYWORDS_FILE)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function
        )
        self.query_constructor = query_constructor
        self.vector_distance_max = 2.0  # 向量检索最大距离（用于归一化）
        
    def jaccard_similarity(self,word1, word2):
        set1 = set(word1)
        set2 = set(word2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union != 0 else 0


    def vector_retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        使用向量检索模型（ChromaDB）进行检索。
        
        :param query: 用户输入的查询字符串
        :param top_k: 返回的Top-K结果数量
        :return: 包含文档ID、距离和元数据的列表
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
        )
        return results
    
    def keywords_search(self, keyword_in_query: str, top_k: int = 1) -> List[Dict]:
        self.keywords["similarity"] = self.keywords["disease_name"].apply(
            lambda x: self.jaccard_similarity(x, keyword_in_query))
        results = self.keywords.sort_values(by="similarity", ascending=False).head(top_k)
        return results.iloc[:top_k][["disease_name", "similarity"]].to_dict('records') 
    
    def keywords_retrieve(self, query: str, top_k: int = 1) -> List[Dict]:
        keywords_in_query = self.query_constructor.extract_category(query)
        if not keywords_in_query or keywords_in_query[0] == '无':
            return {}  # 统一返回字典类型，避免后续判断错误

        or_expression_list = []
        keyword_similarities = {}  # 存储关键词相似度
        for keyword_in_query in keywords_in_query:
            keywords_in_dict = self.keywords_search(keyword_in_query, top_k)
            for item in keywords_in_dict:
                keyword = item["disease_name"]
                keyword_similarities[keyword] = item["similarity"]  # 记录关键词相似度
                or_expression_list.append({"related_disease_1": keyword})
                or_expression_list.append({"related_disease_2": keyword})

        if not or_expression_list:
            return {}

        results = self.collection.get(where={"$or": or_expression_list})
        # 为关键词检索结果添加相似度信息
        if results.get("metadatas"):
            for i, meta in enumerate(results["metadatas"]):
                # 找到匹配的关键词并获取相似度
                matched_keyword = next(
                    (k for k in keyword_similarities if k in [meta.get("related_disease_1"), meta.get("related_disease_2")]),
                    None
                )
                if matched_keyword:
                    # 关键词检索分数：归一化到与向量距离同量级（越高越相关）
                    results.setdefault("similarities", []).append(keyword_similarities[matched_keyword])
                else:
                    results.setdefault("similarities", []).append(0.0)
        return results

    def hybrid_retrieve(self, query: str, top_k: int = 5) -> Dict:
        """优化后的混合检索：统一分数维度，加权融合，严格去重，控制数量"""
        # 1. 并行执行两种检索（实际项目可使用concurrent.futures实现真正并行）
        vector_results = self.vector_retrieve(query, top_k)
        keywords_results = self.keywords_retrieve(query)

        # 2. 标准化向量检索结果格式
        vector_hits = []
        if vector_results.get("ids") and vector_results["ids"][0]:
            for idx, doc_id in enumerate(vector_results["ids"][0]):
                # 向量距离归一化（转为0-1范围，值越小越相关）
                normalized_dist = min(vector_results["distances"][0][idx] / self.vector_distance_max, 1.0)
                vector_hits.append({
                    "id": doc_id,
                    "document": vector_results["documents"][0][idx],
                    "metadata": vector_results["metadatas"][0][idx],
                    "score": 1 - normalized_dist,  # 转为相似度分数（越高越相关）
                    "source": "vector"
                })

        # 3. 标准化关键词检索结果格式
        keyword_hits = []
        if keywords_results.get("ids"):
            for idx, doc_id in enumerate(keywords_results["ids"]):
                # 关键词分数已在keywords_retrieve中计算（0-1范围）
                keyword_hits.append({
                    "id": doc_id,
                    "document": keywords_results["documents"][idx],
                    "metadata": keywords_results["metadatas"][idx],
                    "score": keywords_results["similarities"][idx],
                    "source": "keyword"
                })

        # 4. 融合结果（去重+加权）
        merged = {}
        for hit in vector_hits + keyword_hits:
            doc_id = hit["id"]
            if doc_id not in merged:
                merged[doc_id] = hit
            else:
                # 对重复文档进行分数融合（向量检索权重更高）
                merged[doc_id]["score"] = 0.7 * merged[doc_id]["score"] + 0.3 * hit["score"]
                merged[doc_id]["source"] = "hybrid"

        # 5. 按分数排序并截断到top_k
        sorted_hits = sorted(merged.values(), key=lambda x: x["score"], reverse=True)[:top_k]

        # 6. 转换为标准输出格式
        return {
            "ids": [hit["id"] for hit in sorted_hits],
            "documents": [hit["document"] for hit in sorted_hits],
            "metadatas": [hit["metadata"] for hit in sorted_hits],
            "distances": [1 - hit["score"] for hit in sorted_hits]  # 转回距离格式（越低越相关）
        }


    def chinese_tokenizer(self, text: str) -> List[str]:
        return jieba.lcut(text)
    def bm25_retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        使用BM25模型进行检索，返回包含text、score及metadata中ask和title的结果。
        
        :param query: 用户输入的查询字符串
        :param top_k: 返回的Top-K结果数量
        :return: 包含text、score、ask、title的列表
        """
        results = self.hybrid_retrieve(query, 2 * top_k)
        
        # 准备带完整元数据的文档（保留原始metadata用于后续后提取ask和title）
        documents = []
        for meta in results["metadatas"]:
            # 将完整元数据存入Document的metadata字段
            doc = Document(
                text=meta["answer"],  # 文档内容用answer字段
                metadata={
                          "ask":meta["ask"],
                          "department":meta["department"]
                          }  # 保留原始完整元数据
            )
            documents.append(doc)
        
        # 文档分块（分块后仍会继承原始metadata）
        splitter = SentenceSplitter(chunk_size=1024)
        nodes = splitter.get_nodes_from_documents(documents)
        
        # 初始化BM25检索器
        retriever = BM25Retriever.from_defaults(
            nodes=nodes,
            tokenizer=self.chinese_tokenizer,
            similarity_top_k=top_k
        )
        
        # 执行检索
        bm25_nodes = retriever.retrieve(query)
        
        # 构建最终结果：提取text、score及metadata中的ask和title
        results_with_info = []
        for node in bm25_nodes:
            results_with_info.append({
                "text": node.text,  # 分块后的文本内容
                "similarity":node.score,
                "department":node.metadata.get("department",""),
                "ask": node.metadata.get("ask",""),  # 从元数据提取ask
            })
        return results_with_info
    def retrieve(self, retrieval_type: str, query: str, **kwargs) -> List[Dict]:
        """统一检索接口（调度器）
        Args:
            retrieval_type: 检索类型，可选值：vector/keywords/hybrid/bm25
            query: 用户查询文本
            **kwargs: 其他参数（如top_k，仅vector/hybrid支持）
        Returns:
            检索结果（格式因类型略有差异，见具体方法）
        """
        # 提取通用参数（top_k默认值根据检索类型调整）
        top_k = kwargs.get("top_k", 5)
        # 匹配检索类型，调用对应方法
        if retrieval_type == RetrievalMethod.VECTOR.value:
            query_results = self.vector_retrieve(query, top_k=top_k)
            metadatas = query_results.get("metadatas", [[]])[0]
            distances = query_results.get("distances", [[]])[0]
            
            results=[]
            # 遍历结果，组合需要的字段
            for meta, distance in zip(metadatas, distances):
                filtered_item = {
                    "ask": meta.get("ask", ""),       # 从元数据提取ask
                    "answer": meta.get("answer", ""), # 从元数据提取answer
                    "department": meta.get("department", ""),
                    "similarity": distance              # 加入距离信息
                }
            results.append(filtered_item)
            return results
        elif retrieval_type == RetrievalMethod.HYBRID.value:
            hybrid_results = self.hybrid_retrieve(query, top_k=top_k)
            # 处理混合检索结果格式
            metadatas = hybrid_results.get("metadatas", [])
            distances = hybrid_results.get("distances", [])
            
            results = []
            for meta, distance in zip(metadatas, distances):
                filtered_item = {
                    "ask": meta.get("ask", ""),
                    "answer": meta.get("answer", ""),
                    "department": meta.get("department", ""),
                    "similarity": distance
                }
                results.append(filtered_item)
            return results
        elif retrieval_type == RetrievalMethod.BM25.value:
            return self.bm25_retrieve(query, top_k=top_k)
        else:
            raise ValueError(f"不支持的检索类型：{retrieval_type}，可选值：vector/keywords/hybrid/bm25")
    
if __name__ == "__main__":
    query = "糖尿病的症状有哪些？"
    top_k = 3
    retriever = Retrieval(query_constructor= QueryConstructor())
    query_constructor = QueryConstructor()
    results = retriever.retrieve(RetrievalMethod.BM25.value,query,top_k = 3)
    for i in range(len(results)):
        print(f"{i+1}.{results[i]}\n")
