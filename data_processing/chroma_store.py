import os
import re
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
from tqdm import tqdm
import json  # 用于格式化输出JSON
from zai import ZhipuAiClient
from typing import List

ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")
COLLECTION_NAME = "medical_db"
CHROMA_STORAGE_PATH = "DATA/chroma_db"
INPUT_PATH='processed_data'

class ZhipuEmbeddingFunction:
    """智谱Embedding3嵌入函数"""
    """（添加name属性）"""
    def name(self):
        """返回嵌入模型名称（Chroma要求的方法）"""
        return "zhipu-embedding-3"  # 关键修复：将name改为方法
    def __call__(self, input: List[str]) -> List[List[float]]:
        self.zhipu_api_key = ZHIPU_API_KEY
        if not self.zhipu_api_key:
             raise ValueError("ZHIPU_API_KEY environment variable not set.")
        client = ZhipuAiClient(api_key= self.zhipu_api_key)
        response = client.embeddings.create(
                        model="embedding-3", #填写需要调用的模型编码
                        input=input,
                    )
        return [data_point.embedding for data_point in response.data]

class ChromaStore:
    def __init__(self, collection_name=COLLECTION_NAME, storage_path=CHROMA_STORAGE_PATH,input_path='data_processing/DATA'):
        self.client = chromadb.PersistentClient(path=storage_path)
        self.embedding_function = ZhipuEmbeddingFunction()
        self.input_path = input_path
        self.storage_path = storage_path
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )  
    def load_data(self, file_path):
        """从CSV文件加载数据"""
        df = pd.read_csv(file_path)
        print(f"{file_path} 原始数据行数：{len(df)}")
    
        # 只删除关键列（title/department/answer）为空的行，其他列空值保留
        df = df.dropna(subset=["title", "department", "answer"])
        
        print(f"{file_path} 过滤后数据行数：{len(df)}")
        return df

    def store_data(self, batch_size=64):
        """分批存储数据到Chroma"""
        count = 0
        list_dir = os.listdir(self.input_path) 
        for file_name in list_dir:
            if file_name.endswith('.csv'):
                file_path = os.path.join(self.input_path, file_name)
                df = self.load_data(file_path)
                count += 1
                print(f"文件{file_name}__/{len(list_dir)}加载完成，共{len(df)}条数据")
            
                total_rows = len(df)
                for start_idx in tqdm(range(0, total_rows, batch_size), desc="存储进度"):
                    end_idx = min(start_idx + batch_size, total_rows)
                    batch = df.iloc[start_idx:end_idx]
                    #print(batch)
                    metadatas=batch[["ask", "department", "related_disease_1","related_disease_2","answer"]].to_dict(orient="records")
                    #print(metadatas[0])
                    print(type(metadatas[0]))
                    self.collection.add(
                        documents=batch["title"].tolist(),
                        metadatas=[metadatas[i] for i in range(len(metadatas))],
                        ids=[f"id_{i}" for i in range(start_idx, end_idx)]
                    )
                print(self.collection.get(
                    where={"department": "心血管科"}
                ))
                

if __name__ == "__main__":

    # 初始化→存入Chroma
    store = ChromaStore(
        collection_name=COLLECTION_NAME,
        storage_path=CHROMA_STORAGE_PATH,
        input_path= INPUT_PATH
    )
    store.store_data()
