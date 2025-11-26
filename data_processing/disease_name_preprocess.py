import pandas as pd
import re

COMMON_DISEASE_NAME_LENGTH = 6
def filter_diseases_by_suffix(excel_path):
    print("Reading Excel file...")
    try:
        # Read Excel (by default, it reads the first worksheet; disease names are in the first column by default).
        df_raw = pd.read_excel(excel_path, header=None)  # No table header to avoid column name interference.
        disease_col = df_raw.iloc[:, 2]  # Take the first column as the disease name column.
        print(f"Successfully read {len(disease_col)} raw data entries.")
    except Exception as e:
        print(f"Failed to read Excel: {str(e)}")
        return

    # --------------------------
    # 2. Define the regular expression for disease suffixes
    # --------------------------
    # Core: Cover common disease suffixes (inflammatory/disease/syndrome, etc.) + special disease types to avoid missed detections

    DISEASE_SUFFIX_REGEX = r"""
    ^[a-zA-Z0-9\u4e00-\u9fa5\s\(\)\-\/]+?  # 前缀：允许字母/数字/中文/空格/括号/连接符
    (?:
        # 常见疾病后缀（核心筛选依据）
        综合征|综合症|病|症|炎|瘤|癌|疮|疡|痔|癣|斑|疹|毒|疯|
        障碍|缺陷|变性|增生|硬化|萎缩|畸形|麻痹|贫血|血症|水肿|
        积液|肥大|息肉|结石|囊肿|结核|痢疾|伤寒|
        # 器官特异性疾病后缀（补充筛选，避免漏筛常见疾病）
        肝炎|肺炎|肾炎|胃炎|肠炎|关节炎|神经炎|角膜炎|结膜炎|
        中耳炎|鼻窦炎|咽喉炎|支气管炎|
        # 特殊疾病类型（无典型后缀但属于疾病的名称）
        哮喘|糖尿病|高血压|冠心病|帕金森|阿尔茨海默|艾滋病|
        流感|麻疹|水痘|带状疱疹|梅毒|淋病|疟疾|霍乱
    )
    [\s\(\)\-\/]*$  # 后缀：允许结尾的空格/括号/连接符（如"糖尿病 (2型)"）
    """
    # Compilation regular expressions: Ignore spaces and newlines, case-insensitive, support Chinese characters.
    disease_pattern = re.compile(DISEASE_SUFFIX_REGEX, re.VERBOSE | re.UNICODE | re.IGNORECASE)

    # --------------------------
    # 3. Filter valid disease names
    # --------------------------
    print("Filtering valid disease names...")
    long_name_diseases = []
    short_name_diseases = []
    invalid_records = []

    # The first iteration filters out disease names that match the suffix rules.
    for idx, raw_name in enumerate(disease_col):
        # Preprocessing: Convert to string, remove leading and trailing spaces, remove special punctuation (to avoid formatting interference).
        if pd.isna(raw_name):  # Skip null values
            invalid_records.append({"序号": idx + 1, "原始内容": "空值", "原因": "空数据"})
            continue
        
        # Convert to string and preprocess
        clean_name = str(raw_name).strip()
        # Remove meaningless special characters (such as full-width spaces and tabs).
        clean_name = re.sub(r"[\t\u3000]", "", clean_name)
        
        if len(clean_name) < 2:  # Skip short text (such as single words or empty strings).
            invalid_records.append({"序号": idx + 1, "原始内容": raw_name, "原因": "文本过短（<2字符）"})
            continue
            
        if disease_pattern.match(clean_name):
            if len(clean_name) <= COMMON_DISEASE_NAME_LENGTH:
                short_name_diseases.append(clean_name)
            else:
                long_name_diseases.append(clean_name)
        else:
            invalid_records.append({"序号": idx + 1, "原始内容": raw_name, "原因": "未匹配疾病后缀规则"})

    # --------------------------
    # 4. Deduplication
    # --------------------------
    print("Deduplication is in progress...")
    
    # Short name deduplication (use set for automatic deduplication)
    unique_short_names = list(set(short_name_diseases))
    # Sort (in descending order by length, and alphabetically if the lengths are the same).
    unique_short_names.sort(key=lambda x: (-len(x), x))
    print(f"Filter out {len(unique_short_names)} unique short names of diseases")

    # Deduplication of Long Name Prefixes
    # First, sort long names by length (from shortest to longest). This allows you to prioritize retaining shorter prefixes.

    long_name_diseases.sort(key=lambda x: len(x))
    
    unique_long_names = []
    # Used to store processed prefixes (including short names and long names that have been retained).
    processed_prefixes = set(unique_short_names)
    
    for long_name in long_name_diseases:
        # Check if the current long name begins with any processed prefix.
        is_duplicate = False
        for prefix in processed_prefixes:
            if long_name.startswith(prefix):
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_long_names.append(long_name)
            processed_prefixes.add(long_name[COMMON_DISEASE_NAME_LENGTH])
    print(f"Filter out {len(unique_long_names)} unique long names of diseases")
    return unique_long_names + unique_short_names
            
        
def add_custom_diseases(csv_path, custom_diseases_csv_path):
    # Load the files
    try:
        df_orig = pd.read_csv(csv_path)
        df_supp = pd.read_csv(custom_diseases_csv_path)

        # Concatenate
        df_combined = pd.concat([df_orig, df_supp])

        # Drop duplicates based on 'disease_name', keeping the first occurrence
        # Assuming the user wants to keep the original if it exists, or the new one if not.
        # Since we are just merging and sorting, order of drop doesn't matter much if names are identical.
        df_combined = df_combined.drop_duplicates(subset='disease_name')

        # Ensure 'name_length' is correct (recalculate just in case)
        df_combined['name_length'] = df_combined['disease_name'].astype(str).apply(len)

        # Sort by name_length in descending order
        df_combined = df_combined.sort_values(by='name_length', ascending=False)

        df_combined.to_csv(csv_path, index=False)

        print(f"Successfully merged and sorted. Total entries: {len(df_combined)}")
        print(df_combined.head())

    except Exception as e:
        print(f"An error occurred: {e}")
        
                 
    
#Objective: To generate a jieba disease dictionary, enabling the use of the jieba library to match and identify diseases in sentences.
if __name__ == "__main__":
    # The data from the Hebei Provincial Health Commission contains a large number of diseases that are not considered diseases by the average person, therefore, the data needs to be filtered.
    # Source: Hebei Provincial Health Commission (xlsx file)
    INPUT_EXCEL = "disease_names.xlsx"
    # Output path: CSV file of filtered diseases (customizable name)
    OUTPUT_CSV = "./disease_names_processed.csv"

    # Perform filtering
    res = filter_diseases_by_suffix(INPUT_EXCEL)
    df_final = pd.DataFrame({
        "disease_name": res,
        "name_length": [len(name) for name in res],
    })
    
    # Sort by length in descending order; sort by alphabetical order if lengths are the same.
    df_final = df_final.sort_values(by=["name_length", "disease_name"], ascending=[False, True]).reset_index(drop=True)
    
    # Save as CSV
    df_final.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"The filtering results have been saved to {OUTPUT_CSV}, containing a total of {len(df_final)} valid disease names.")

    # The filtered data does not contain many common disease names
    # Therefore, it is merged with the common disease list (supplementary_common_diseases.csv) generated by genai as a supplement.
    add_custom_diseases(OUTPUT_CSV, 'supplementary_common_diseases.csv')
    
    
    