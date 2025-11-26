import os
import re
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
from tqdm import tqdm
import json  # Used for formatting JSON output
from zai import ZhipuAiClient
from typing import List

_ZHIPU_KEY_FILE = Path("data_processing/API_KEY.txt")


def _load_zhipu_api_key() -> str:
    try:
        if _ZHIPU_KEY_FILE.exists():
            return _ZHIPU_KEY_FILE.read_text(encoding="utf-8").strip()
    except Exception:
        # Ignore file read errors and retrieve the file from environment variables instead.
        pass
    return os.environ.get("ZHIPU_API_KEY", "")


ZHIPU_API_KEY = _load_zhipu_api_key()
COLLECTION_NAME = "medical_db"
CHROMA_STORAGE_PATH = "data_processing/DATA/chroma_db"
INPUT_PATH = "data_processing/processed_data"


class ZhipuEmbeddingFunction:
    """ZhipuEmbedding3 embedding function"""

    """(Add name attribute)"""

    def name(self):
        """Returns the name of the embedded model (the method required by Chroma)."""
        return "zhipu-embedding-3"  # Critical fix: Change name to method

    def __call__(self, input: List[str]) -> List[List[float]]:
        self.zhipu_api_key = ZHIPU_API_KEY
        if not self.zhipu_api_key:
            raise ValueError("ZHIPU_API_KEY environment variable not set.")
        client = ZhipuAiClient(api_key=self.zhipu_api_key)
        response = client.embeddings.create(
            model="embedding-3",  # Enter the model code to be called.
            input=input,
        )
        return [data_point.embedding for data_point in response.data]


class ChromaStore:
    def __init__(
        self,
        collection_name=COLLECTION_NAME,
        storage_path=CHROMA_STORAGE_PATH,
        input_path="data_processing/DATA",
    ):
        self.client = chromadb.PersistentClient(path=storage_path)
        self.embedding_function = ZhipuEmbeddingFunction()
        self.input_path = input_path
        self.storage_path = storage_path
        self.collection = self.client.get_or_create_collection(
            name=collection_name, embedding_function=self.embedding_function
        )

    def load_data(self, file_path):
        """Load data from a CSV file"""
        df = pd.read_csv(file_path)
        print(f"{file_path} Original number of rows: {len(df)}")
    
        # Delete only rows where the key columns (title/department/answer) are empty; retain empty values in other columns.
        df = df.dropna(subset=["title", "department", "answer"])
        
        print(f"Number of rows after filtering by {file_path}: {len(df)}")
        return df

    def store_data(self, batch_size=64):
        """Store data to Chroma in batches"""
        count = 0
        list_dir = os.listdir(self.input_path)
        for file_name in list_dir:
            if file_name.endswith(".csv"):
                
                file_path = os.path.join(self.input_path, file_name)
                df = self.load_data(file_path)
                count += 1
                print(f"file {file_name}__/{len(list_dir)} loading successfully，total {len(df)} data")

                total_rows = len(df)
                file_name_prefix, _ = os.path.splitext(file_name)
                for start_idx in tqdm(
                    range(0, total_rows, batch_size), desc="存储进度"
                ):
                    end_idx = min(start_idx + batch_size, total_rows)
                    batch = df.iloc[start_idx:end_idx]
                    # print(batch)
                    metadatas = batch[
                        [
                            "ask",
                            "department",
                            "related_disease_1",
                            "related_disease_2",
                            "answer",
                        ]
                    ].to_dict(orient="records")
                    # print(metadatas[0])
                    print(type(metadatas[0]))
                    self.collection.add(
                        documents=batch["title"].tolist(),
                        metadatas=[metadatas[i] for i in range(len(metadatas))],
                        ids=[f"{file_name_prefix}_{i}" for i in range(start_idx, end_idx)],
                    )
                print(self.collection.get(where={"department": "心血管科"}))


if __name__ == "__main__":

    # Initialize → Store in Chroma
    store = ChromaStore(
        collection_name=COLLECTION_NAME,
        storage_path=CHROMA_STORAGE_PATH,
        input_path=INPUT_PATH,
    )
    store.store_data()
