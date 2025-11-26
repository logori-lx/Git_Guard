import os
import re
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
from tqdm import tqdm

# Configuration parameters
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"  # Disable symbolic link warnings
MEDICAL_DATA_DIR = r"D:\software\software engineering\project"  # Data root directory
BGE_MODEL_NAME = "BAAI/bge-small-zh-v1.5"  # Chinese medical embedding model
COLLECTION_NAME = "medical_qa_chroma"  # Chroma collection name
CHROMA_STORAGE_PATH = "./chroma_medical_data"  # Vector data storage path

# Filename and Department Mapping
FILE_DEPARTMENT_MAP = {
    "男科5-13000.csv": "Andriatria_男科",
    "内科5000-33000.csv": "IM_内科",
    "妇产科6-28000.csv": "OAGD_妇产科",
    "儿科5-14000.csv": "Pediatric_儿科",
    "肿瘤科5-10000.csv": "Oncology_肿瘤科",
    "外科5-14000.csv": "Surgical_外科"
}

# Disease Extraction Regularity
DISEASE_PATTERN = re.compile(
    r'((高|低|急|慢|重|轻|先|后|原|继|良|恶)?[\u4e00-\u9fa5]{2,15}?(?:病|症|炎|综合征|瘤|癌|疮|中毒|感染|障碍|缺损|畸形|麻痹|痉挛|出血|梗死|硬化|萎缩|增生|结石|溃疡|疝|脓肿|积液|热|痛|癣|疹|瘫|疸|盲|聋|痹|痨|痢|癣|疣|痔))',
    re.IGNORECASE
)



# Loading medical Q&A data
def load_medical_data(data_dir):
    all_dfs = []
    print(f"Start loading medical Q&A data")
    for filename, department in FILE_DEPARTMENT_MAP.items():
        file_path = Path(data_dir) / filename
        print(f"Loading {filename}...")

        # Specify the encoding and error handling using the open function, then pass it to read_csv.
        with open(file_path, "r", encoding="gbk", errors="ignore") as f:
            df = pd.read_csv(f)

        # Uniform column naming
        df.columns = ["department", "title", "ask", "answer"]
        all_dfs.append(df)
        print(f"Loading {department}：{len(df)} data")

    # Merge all data and reset indexes
    merged_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nData loading complete. Total number of records after merging:{len(merged_df)}")
    return merged_df



# Data cleaning
def clean_medical_data(df):
    print("\nStart data cleaning...")

    # Convert Chinese double quotation marks to English double quotation marks
    def fix_quotes(text):
        if pd.isna(text):
            return text
        return str(text).replace('“', '"').replace('”', '"').strip()

    # Apply to all text columns
    text_cols = ["title", "ask", "answer", "department"]
    for col in text_cols:
        df[col] = df[col].apply(fix_quotes)

    # Handling null values: Delete core fields (ask/answer) if they are empty; fill other fields with None.
    df = df.dropna(subset=["ask", "answer"])  # Delete invalid lines that contain no question or answer content.
    df = df.fillna("None")  # Fill other empty columns with None.
    df = df.replace("", "None")  # Convert an empty string to None

    # Deduplication
    df = df.drop_duplicates(subset=["ask", "answer"], keep="first")

    print(f"Cleaning complete. Remaining valid data: {len(df)} records.")
    return df



# 3. Extract related diseases
def extract_related_diseases(df):
    print("\nStart extracting relevant diseases (optimized regular expressions to cover more disease types)...")

    def extract_disease_from_text(row):
        # Merge title, question, and answer text
        combined_text = f"{row['title']} {row['ask']} {row['answer']}"
        # Regular expression matching for diseases
        diseases = DISEASE_PATTERN.findall(combined_text)
        # Extract disease names from the matching results, remove duplicates and filter noise.
        unique_diseases = list(set([d[0] for d in diseases if len(d[0]) >= 2 and not d[0].endswith("小病")]))
        # Returns a default value if no match is found.
        return unique_diseases if unique_diseases else ["无明确相关疾病"]

    # Batch Extraction
    tqdm.pandas(desc="疾病提取进度")
    df["related_disease"] = df.progress_apply(extract_disease_from_text, axis=1)

    print("Disease extraction completed")
    return df


# Initialize the Chroma vector database
def init_chroma():
    print("\nInitialize the Chroma vector database...")
    # Create a persistent client
    client = chromadb.PersistentClient(path=CHROMA_STORAGE_PATH)

    # Configure Chinese embedding function（BGE-small-zh-v1.5）
    bge_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=BGE_MODEL_NAME
    )

    # Create/Get Collections
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=bge_ef,
        metadata={"description": "医疗问答数据集"}
    )

    print(f"Chroma initialization complete, collection name: {COLLECTION_NAME}")
    return collection



# 5. Store in the Chroma vector database (convert list to string, adapt to metadata format).
def save_to_chroma(collection, df):
    print("\nStart storing data into Chroma...")

    # Required format for constructing Chroma：ids、documents、metadatas
    ids = [str(i + 1) for i in range(len(df))]  # ID starts from 1 and increments automatically.
    documents = df["answer"].tolist()  # Doctor's answers serve as the core content of the search.
    metadatas = [
        {
            "department": row["department"],
            "related_disease": ",".join(row["related_disease"]),  # list to string
            "title": row["title"],
            "query": row["ask"]  # User questions are stored in metadata for easy viewing.
        }
        for _, row in df.iterrows()
    ]

    # Deposit in batches
    batch_size = 1000
    for i in tqdm(range(0, len(ids), batch_size), desc="存入进度"):
        batch_ids = ids[i:i + batch_size]
        batch_docs = documents[i:i + batch_size]
        batch_metas = metadatas[i:i + batch_size]
        collection.add(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_metas
        )

    print(f"Data storage complete. Total number of records in the collection:{collection.count()}")


# Query from Chroma
def query_chroma(collection, query_text, top_k=3):
    """The search query returns the most similar question-and-answer pairs to medical questions, and outputs results in standard Chinese."""
    results = collection.query(
        query_texts=[query_text],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]  # No need to explicitly specify ids
    )

    # Results: String converted back to list, adapted to display format.
    formatted_results = []
    for i in range(len(results["ids"][0])):
        # Disease string converted back to list
        related_diseases = results["metadatas"][0][i]["related_disease"].split(",")
        formatted_results.append({
            "id": results["ids"][0][i],
            "related_disease": related_diseases,
            "department": results["metadatas"][0][i]["department"],
            "user_query": results["metadatas"][0][i]["query"],
            "doctor_answer": results["documents"][0][i],
            "similarity": 1 - results["distances"][0][i]  # Distance to Similarity
        })
    return formatted_results

if __name__ == "__main__":
    if Path(CHROMA_STORAGE_PATH).exists():
        print(f"⚠️ Old data folder {CHROMA_STORAGE_PATH} found. Please delete it manually before running!")
        print("(Old data contains garbled characters or encoding errors; it must be cleared before new data can be stored.)")
        exit()

    # Load merged data
    medical_df = load_medical_data(MEDICAL_DATA_DIR)

    # Data cleaning (removing quotation marks, null values, and duplicates)
    cleaned_df = clean_medical_data(medical_df)

    # Extract relevant diseases (regular expression)
    final_df = extract_related_diseases(cleaned_df)

    # Initialize the Chroma vector database
    chroma_collection = init_chroma()

    # Store in Chroma
    save_to_chroma(chroma_collection, final_df)

    # Query verification
    test_queries = [
        "高血压患者能吃党参吗？",
        "儿科宝宝发烧怎么护理？",
        "妇产科月经不调的原因有哪些？",
        "肿瘤科癌症患者饮食需要注意什么？"
    ]

    # Execute the test query and print the results.
    for query in test_queries:
        print("\n" + "=" * 60)
        print(f"query problem：{query}")
        print("Most similar medical Q&A results：")
        results = query_chroma(chroma_collection, query, top_k=2)

        for i, res in enumerate(results, 1):
            print(f"\nThe {i}th item (similarity:{res['similarity']:.4f}）")
            print(f"Department：{res['department']}")
            print(f"Related diseases：{', '.join(res['related_disease'])}")
            print(f"User Questions：{res['user_query']}")
            print(f"Doctor's answer：{res['doctor_answer'][:100]}...")  # Extract the first 100 characters to avoid outputting too long.