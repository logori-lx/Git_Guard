import os
import chromadb
from typing import List, Dict
from pyparsing import Enum
from zai import ZhipuAiClient  # 确保 zai.py 在PYTHONPATH中
import pandas as pd
from query_constructing.query_constructor import QueryConstructor
from llama_index.core import Document
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.node_parser import SentenceSplitter
import jieba
import sys


# 获取项目根目录的绝对路径（当前文件在 retrieve 文件夹下，上一级就是根目录）
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# 将根目录加入 Python 系统路径
sys.path.append(root_path)
API_KEY = os.getenv("MEDICAL_RAG")
COLLECTION_NAME = "medical_db"
# 解决路径问题
KEYWORDS_FILE = os.path.join(root_path,"data_processing","DATA","related_disease","disease_names_processed.csv")
DB_PATH = os.path.join(root_path,"data_processing","DATA","chroma_db")


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
            n_results=top_k
        )
        return results
    
    def keywords_search(self, keyword_in_query: str, top_k: int = 1) -> List[Dict]:
        """
        使用关键词检索模型（BM25）进行检索。
        
        :param query: 用户输入的查询字符串
        :param top_k: 返回的Top-K结果数量
        :return: 包含文档ID、距离和元数据的列表
        """
        
        # 让keywords与self.keywords进行相似度比较，取相似度最高的top_k个
        self.keywords["similarity"] = self.keywords["disease_name"].apply(lambda x: self.jaccard_similarity(x, keyword_in_query))
        results = self.keywords.sort_values(by="similarity", ascending=False).head(top_k)
        return results.iloc[:top_k]["disease_name"].tolist()
    
    def keywords_retrieve(self, query: str, top_k:int = 1) -> List[Dict]:
        """
        使用关键词检索模型（BM25）进行检索。
        
        :param query: 用户输入的查询字符串
        :param top_k: 返回的Top-K结果数量
        :return: 包含文档ID、距离和元数据的列表
        """
        keywords_in_query = self.query_constructor.extract_category(query)
        if len(keywords_in_query) == 0 or keywords_in_query[0] == '无':
            return False
        or_expression_list = []
        for keyword_in_query in keywords_in_query:
            keywords_in_dict = self.keywords_search(keyword_in_query, top_k)
            for keyword in keywords_in_dict:
                or_expression_list.append({"related_disease_1": keyword})
                or_expression_list.append({"related_disease_2": keyword})
        results = self.collection.get(
            where={
                "$or":or_expression_list
            }
        )
        return results
    
    def hybrid_retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        使用混合检索模型（向量检索 + 关键词检索）进行检索。
        
        :param query: 用户输入的查询字符串
        :param top_k: 返回的Top-K结果数量
        :return: 包含文档ID、距离和元数据的列表
        """
        vector_results = self.vector_retrieve(query, top_k)
        keywords_results = self.keywords_retrieve(query)
        #print(f"vector_results metadatas: {vector_results['metadatas']}")
        results = {
            'ids': vector_results['ids'],
            'documents': vector_results['documents'][0],
            'metadatas': vector_results['metadatas'][0],
            'distances': vector_results['distances'][0]

        }
        #print(f"results metadatas: {results['metadatas']}")
        if not keywords_results:
            return results
        else:
            for i in range(len(keywords_results["ids"])):
                if keywords_results["ids"][i] in results["ids"]:
                    continue
                else:
                    results["ids"].append(keywords_results["ids"][i])
                    results["documents"].append(keywords_results["documents"][i])
                    results["metadatas"].append(keywords_results["metadatas"][i])
                    results["distances"].append(10000) # 由于不适用所以赋一个巨大的值
        return results

    def chinese_tokenizer(self, text: str) -> List[str]:
        return jieba.lcut(text)
    def bm25_retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        使用BM25模型进行检索。
        
        :param query: 用户输入的查询字符串
        :param top_k: 返回的Top-K结果数量
        :return: 包含文档ID、距离和元数据的列表
        """
        results = self.hybrid_retrieve(query, 2*top_k)
        #print(f"hybrid_retrieve results['metadatas']: {results['metadatas']}")
        # 准备示例文档
        documents = []
        for i in results["metadatas"]:
            documents.append(Document(text=i["answer"]))
        # 文档分块（可选，根据需求调整）
        splitter = SentenceSplitter(chunk_size=1024)
        nodes = splitter.get_nodes_from_documents(documents)
        # 初始化 BM25Retriever
        retriever = BM25Retriever.from_defaults(
            nodes=nodes,  # 传入分块后的节点或直接传入 documents
            tokenizer=self.chinese_tokenizer,
            similarity_top_k=top_k  # 检索返回的Top K结果数量
        )

        # 执行检索
        results_bm25 = retriever.retrieve(query)
        return results_bm25

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
        rewritten_query = self.query_constructor.get_query(query)
        # 匹配检索类型，调用对应方法
        if retrieval_type == RetrievalMethod.VECTOR.value:
            return self.vector_retrieve(rewritten_query, top_k=top_k)
        elif retrieval_type == RetrievalMethod.HYBRID.value:
            return self.hybrid_retrieve(rewritten_query, top_k=top_k)
        elif retrieval_type == RetrievalMethod.BM25.value:
            # BM25的top_k单独控制（默认3）
            bm25_top_k = kwargs.get("bm25_top_k", 3)
            return self.bm25_retrieve(rewritten_query, top_k=bm25_top_k)
        else:
            raise ValueError(f"不支持的检索类型：{retrieval_type}，可选值：vector/keywords/hybrid/bm25")
    
if __name__ == "__main__":
    query = "糖尿病的症状有哪些？"
    top_k = 3
    retriever = Retrieval(query_constructor= QueryConstructor())
    query_constructor = QueryConstructor()
    results = retriever.retrieve(RetrievalMethod.BM25.value,query,top_k = 3)
    results_final = [i.text for i in results]
    for i in range(len(results_final)):
        print(f"{i+1}.______ {results_final[i]}\n")
