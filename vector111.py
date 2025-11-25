import os
import re
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
from tqdm import tqdm

# 配置参数
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"  # 禁用符号链接警告
MEDICAL_DATA_DIR = r"D:\software\software engineering\project"  # 数据根目录
BGE_MODEL_NAME = "BAAI/bge-small-zh-v1.5"  # 中文医疗嵌入模型
COLLECTION_NAME = "medical_qa_chroma"  # Chroma集合名
CHROMA_STORAGE_PATH = "./chroma_medical_data"  # 向量数据存储路径

# 文件名与科室映射
FILE_DEPARTMENT_MAP = {
    "男科5-13000.csv": "Andriatria_男科",
    "内科5000-33000.csv": "IM_内科",
    "妇产科6-28000.csv": "OAGD_妇产科",
    "儿科5-14000.csv": "Pediatric_儿科",
    "肿瘤科5-10000.csv": "Oncology_肿瘤科",
    "外科5-14000.csv": "Surgical_外科"
}

# 疾病提取正则
DISEASE_PATTERN = re.compile(
    r'((高|低|急|慢|重|轻|先|后|原|继|良|恶)?[\u4e00-\u9fa5]{2,15}?(?:病|症|炎|综合征|瘤|癌|疮|中毒|感染|障碍|缺损|畸形|麻痹|痉挛|出血|梗死|硬化|萎缩|增生|结石|溃疡|疝|脓肿|积液|热|痛|癣|疹|瘫|疸|盲|聋|痹|痨|痢|癣|疣|痔))',
    re.IGNORECASE
)



# 加载医疗问答数据
def load_medical_data(data_dir):
    all_dfs = []
    print(f"开始加载医疗问答数据")
    for filename, department in FILE_DEPARTMENT_MAP.items():
        file_path = Path(data_dir) / filename
        print(f"正在加载 {filename}...")

        # 用open函数指定编码和错误处理，再传给read_csv
        with open(file_path, "r", encoding="gbk", errors="ignore") as f:
            df = pd.read_csv(f)

        # 统一列名
        df.columns = ["department", "title", "ask", "answer"]
        all_dfs.append(df)
        print(f"加载 {department}：{len(df)} 条数据")

    # 合并所有数据并重置索引
    merged_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\n数据加载完成，合并后总条数：{len(merged_df)}")
    return merged_df



# 数据清洗
def clean_medical_data(df):
    print("\n开始数据清洗...")

    # 中文双引号转英文双引号
    def fix_quotes(text):
        if pd.isna(text):
            return text
        return str(text).replace('“', '"').replace('”', '"').strip()

    # 应用于所有文本列
    text_cols = ["title", "ask", "answer", "department"]
    for col in text_cols:
        df[col] = df[col].apply(fix_quotes)

    # 处理空值：核心字段（ask/answer）为空则删除，其他字段填充None
    df = df.dropna(subset=["ask", "answer"])  # 删除无问答内容的无效行
    df = df.fillna("None")  # 其他空列填充None
    df = df.replace("", "None")  # 空字符串转为None

    # 去重
    df = df.drop_duplicates(subset=["ask", "answer"], keep="first")

    print(f"清洗完成，剩余有效数据：{len(df)} 条")
    return df



# 3. 提取相关疾病
def extract_related_diseases(df):
    print("\n开始提取相关疾病（优化正则，覆盖更多疾病类型）...")

    def extract_disease_from_text(row):
        # 合并标题、提问、回答文本
        combined_text = f"{row['title']} {row['ask']} {row['answer']}"
        # 正则匹配疾病
        diseases = DISEASE_PATTERN.findall(combined_text)
        # 提取匹配结果中的疾病名称，去重+过滤噪声
        unique_diseases = list(set([d[0] for d in diseases if len(d[0]) >= 2 and not d[0].endswith("小病")]))
        # 无匹配结果时返回默认值
        return unique_diseases if unique_diseases else ["无明确相关疾病"]

    # 批量提取
    tqdm.pandas(desc="疾病提取进度")
    df["related_disease"] = df.progress_apply(extract_disease_from_text, axis=1)

    print("疾病提取完成")
    return df


# 初始化Chroma向量数据库
def init_chroma():
    print("\n初始化Chroma向量数据库...")
    # 创建持久化客户端
    client = chromadb.PersistentClient(path=CHROMA_STORAGE_PATH)

    # 配置中文嵌入函数（BGE-small-zh-v1.5）
    bge_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=BGE_MODEL_NAME
    )

    # 创建/获取集合
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=bge_ef,
        metadata={"description": "医疗问答数据集"}
    )

    print(f"Chroma初始化完成，集合名：{COLLECTION_NAME}")
    return collection



# 5. 存入Chroma向量数据库（列表转字符串，适配元数据格式）
def save_to_chroma(collection, df):
    print("\n开始向Chroma存入数据...")

    # 构造Chroma所需格式：ids、documents、metadatas
    ids = [str(i + 1) for i in range(len(df))]  # ID从1开始自增
    documents = df["answer"].tolist()  # 医生回答作为检索核心内容
    metadatas = [
        {
            "department": row["department"],
            "related_disease": ",".join(row["related_disease"]),  # 列表转字符串
            "title": row["title"],
            "query": row["ask"]  # 用户提问存入元数据，方便查看
        }
        for _, row in df.iterrows()
    ]

    # 分批存入
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

    print(f"数据存入完成，集合中总条数：{collection.count()}")


# 从Chroma查询
def query_chroma(collection, query_text, top_k=3):
    """查询与医疗问题最相似的问答对，返回正常中文结果"""
    results = collection.query(
        query_texts=[query_text],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]  # 无需显式指定ids
    )

    # 整理结果：字符串转回列表，适配显示格式
    formatted_results = []
    for i in range(len(results["ids"][0])):
        # 疾病字符串转回列表
        related_diseases = results["metadatas"][0][i]["related_disease"].split(",")
        formatted_results.append({
            "id": results["ids"][0][i],
            "related_disease": related_diseases,
            "department": results["metadatas"][0][i]["department"],
            "user_query": results["metadatas"][0][i]["query"],
            "doctor_answer": results["documents"][0][i],
            "similarity": 1 - results["distances"][0][i]  # 距离转相似度
        })
    return formatted_results

if __name__ == "__main__":
    if Path(CHROMA_STORAGE_PATH).exists():
        print(f"⚠️  发现旧数据文件夹 {CHROMA_STORAGE_PATH}，请手动删除后再运行！")
        print("   （旧数据含乱码或编码异常，必须清空才能存储新数据）")
        exit()

    # 加载合并数据
    medical_df = load_medical_data(MEDICAL_DATA_DIR)

    # 数据清洗（处理引号、空值、去重）
    cleaned_df = clean_medical_data(medical_df)

    # 提取相关疾病（正则）
    final_df = extract_related_diseases(cleaned_df)

    # 初始化Chroma向量数据库
    chroma_collection = init_chroma()

    # 存入Chroma
    save_to_chroma(chroma_collection, final_df)

    # 查询验证
    test_queries = [
        "高血压患者能吃党参吗？",
        "儿科宝宝发烧怎么护理？",
        "妇产科月经不调的原因有哪些？",
        "肿瘤科癌症患者饮食需要注意什么？"
    ]

    # 执行测试查询并打印结果
    for query in test_queries:
        print("\n" + "=" * 60)
        print(f"查询问题：{query}")
        print("最相似的医疗问答结果：")
        results = query_chroma(chroma_collection, query, top_k=2)

        for i, res in enumerate(results, 1):
            print(f"\n第{i}条（相似度：{res['similarity']:.4f}）")
            print(f"科室：{res['department']}")
            print(f"相关疾病：{', '.join(res['related_disease'])}")
            print(f"用户提问：{res['user_query']}")
            print(f"医生回答：{res['doctor_answer'][:100]}...")  # 截取前100字，避免输出过长