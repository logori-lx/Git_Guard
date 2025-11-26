import pandas as pd
import jieba
import os

JIEBA_DICT_PATH = "./jieba_dict.txt"

class DiseaseExtractor:
    def __init__(self, disease_dict_csv):
        """
        Initialize the disease extractor, loading disease names to build a Jieba custom dictionary.

        :param disease_dict_csv: The path to the CSV file containing disease names (e.g., your disease_names_processed.csv)
        """
        if not os.path.exists(JIEBA_DICT_PATH):
            print("Loading disease names and building the Jieba dictionary...")
            # Read the disease CSV (assuming the disease name is in the "disease_name" column; if the column name is different, you need to modify it).
            disease_df = pd.read_csv(disease_dict_csv, encoding="utf-8-sig")
            # Extract unique disease names (to remove duplicates and avoid dictionary redundancy).
            disease_names = disease_df["disease_name"].dropna().str.strip().unique().tolist()
            
            # Generate a custom Jieba dictionary (format: word frequency part of speech, set frequency to 1000 to ensure priority word segmentation).
            jieba_dict_content = "\n".join([f"{name} 1000 disease" for name in disease_names])
            # Temporarily save dictionary files (Jieba requires loading via file).
            
            with open(JIEBA_DICT_PATH, "w", encoding="utf-8") as f:
                f.write(jieba_dict_content)
            print(f"The Jieba dictionary has been generated, containing {len(disease_names)} disease names.")
        else:
            # If the dictionary file already exists, directly read the list of disease names to initialize the set.
            disease_df = pd.read_csv(disease_dict_csv, encoding="utf-8-sig")
            disease_names = disease_df["disease_name"].dropna().str.strip().unique().tolist()
        
        # Load a custom dictionary (overriding Jieba's default segmentation to ensure disease names are correctly segmented).
        jieba.load_userdict(JIEBA_DICT_PATH)
        print(f"Successfully loaded {len(disease_names)} disease names into the Jieba dictionary.")
        self.disease_set = set(disease_names)
    def extract_diseases_from_text(self, text):
        if pd.isna(text):
            return []
        # Word segmentation (using a custom dictionary to ensure disease names are not split)
        words = jieba.lcut(str(text).strip())
        # Filter out words from the disease set (use a set to speed up the search).
        return [word for word in words if word in self.disease_set]

if __name__ == "__main__":
    # Test code
    extractor = DiseaseExtractor("disease_names_processed.csv")
    sample_text = "患者患有高血压和糖尿病，伴有冠心病史。"
    extracted_diseases = extractor.extract_diseases_from_text(sample_text)
    print("Extracted disease name：", extracted_diseases)
    
    
    