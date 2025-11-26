import pandas as pd
import os
from jieba_extract_disease import DiseaseExtractor  # 导入自定义的疾病提取器

MAX_RELATED_DISEASES_COUNT = 2

class DataPreprocessor:
    def __init__(self, disease_dict_csv):
        """
        Initialize the data preprocessor and load disease names to build a Jieba custom dictionary.

        :param disease_dict_csv: The path to the CSV file containing disease names (e.g., your disease_names_processed.csv)
        """
        self.extractor = DiseaseExtractor(disease_dict_csv)
    
    def process_medical_data(self, input_csv, output_csv):
        # Read input CSV file
        df = pd.read_csv(input_csv, encoding="utf-8-sig")
        
        # Store the Top N diseases of valid rows and the original row data.
        valid_rows = []
        top_diseases_per_row = []
        
        for _, row in df.iterrows():
            # Merge three columns of text content
            text_parts = [
                str(row['department']),
                str(row['title']),
                str(row['ask'])
            ]
            combined_text = ' '.join(text_parts)
            
            # Extract disease name
            diseases = self.extractor.extract_diseases_from_text(combined_text)
            
            # Only process data that identifies diseases
            if not diseases:
                continue  # Skip the current line if there is no disease.

            # Calculate the Top N diseases in the current row
            # Statistical frequency
            freq = {}
            for disease in diseases:
                freq[disease] = freq.get(disease, 0) + 1
            # Sort by frequency (descending order), and if frequencies are the same, sort by order of appearance.
            sorted_diseases = sorted(freq.items(), key=lambda x: (-x[1], diseases.index(x[0])))
            sorted_names = [item[0] for item in sorted_diseases]
            # Take the first N strings; if there are not enough, fill them with empty strings.
            row_top = sorted_names[:MAX_RELATED_DISEASES_COUNT]
            if len(row_top) < MAX_RELATED_DISEASES_COUNT:
                row_top += ["无"] * (MAX_RELATED_DISEASES_COUNT - len(row_top))
            
            # Save valid row data and corresponding diseases
            valid_rows.append(row)
            top_diseases_per_row.append(row_top)
        
        # Create a new DataFrame based on valid rows.
        result_df = pd.DataFrame(valid_rows)
        
        # Add Top N disease columns
        for i in range(MAX_RELATED_DISEASES_COUNT):
            col_name = f'related_disease_{i+1}'
            result_df[col_name] = [row_top[i] for row_top in top_diseases_per_row]
        
        # 保存结果到新CSV
        result_df.to_csv(output_csv, encoding="utf-8-sig", index=False)
        print(f"Processing complete. A total of {len(result_df)} valid data entries were retained. The results have been saved to {output_csv}")
    
    def preprocess_dir(self, process_dir_path, output_dir_path):
        import os
        # Ensure the output directory exists.
        os.makedirs(output_dir_path, exist_ok=True)
        list_dir = os.listdir(process_dir_path) 
        for file_name in list_dir:
            if file_name.endswith('.csv'):
                file_path = os.path.join(process_dir_path, file_name)
                self.process_medical_data(
                    input_csv=file_path,
                    output_csv=os.path.join(output_dir_path, file_name)
                )

if __name__ == "__main__":
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    DISEASE_DICT = os.path.join(root_path, "related_disease","disease_names_processed.csv") 
    preprocessor = DataPreprocessor(disease_dict_csv=DISEASE_DICT)
    preprocessor.preprocess_dir("original_data", "processed_data")