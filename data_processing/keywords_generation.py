import pandas as pd
from zai import ZhipuAiClient
import os
API_KEY = os.getenv("MEDICAL_RAG")

class keywords_generator:
    def __init__(self, api_key=API_KEY):
        self.client = ZhipuAiClient(api_key=api_key)
    def generate(self, query):
        response = self.client.chat.completions.create(
            model="GLM-4.5-AirX",
            messages=[
                {
                    "role": "system",
                    "content": "你是一个医学专家。你会根据用户的话，提炼出与之相关的疾病。"\
                                "如下面两个例子："\
                                "用户：想知道癫痫长期用药的危害有什么"\
                                "你：癫痫"\
                                "用户：有高血压的人能献血吗？"\
                                "你：高血压"\
                                "如果用户没有提到疾病，你应当输出：无"\
                                "如果用户提到了多种疾病，你应当输出最符合的2个疾病名称，用|隔开。"\
                                "请你只输出疾病名称，不要输出其他内容。不允许添加除|以外的其他标点符号。"\
                                
                },
                {
                    "role": "user",
                    "content": query
                }
            ],
            temperature=0.6
            )
        content = response.choices[0].message.content.strip()
        content = content.replace("\n","")
        list_content = content.split("|")
        list_content = [d.strip() for d in list_content]
        if len(list_content) != 0:
            return list_content
        else:
            return ["无"]
# 为csv文件添加related_disease列
class csv_processor:
    def __init__(self, file_path, api_key):
        self.file_path = file_path
        self.client = keywords_generator(api_key)
        self.related_disease = set()
        self.related_disease_file = os.path.join(self.file_path, "related_disease/related_disease.csv")
    def process(self,file_path):
        df = pd.read_csv(file_path)
        # 去除空缺值
        df = df.dropna()
        for i in range(len(df)):
            print(f"正在处理第{i+1}/{len(df)}行")
            disease = self.client.generate(df.loc[i,'title'])
            if disease != ["无"]:
                df.loc[i,"related_disease_1"] = disease[0]
                if len(disease)>1:
                    df.loc[i,"related_disease_2"] = disease[1]
                else:
                    df.loc[i,"related_disease_2"] = '无'
            else:
                df.loc[i,"related_disease_1"] = '无'
                df.loc[i,"related_disease_2"] = '无'
            for d in disease:
                self.related_disease.add(d)
        df.to_csv(file_path, index=False,encoding='utf-8-sig')
    def process_all(self):
        # 处理所有csv文件
        list_dir = os.listdir(self.file_path)
        count = 0
        for file in list_dir:
            if file.endswith(".csv"):
                count += 1
                print(f"正在处理文件：{file}，已处理文件数：{count}")
                self.process(os.path.join(self.file_path, file))
        # 保存所有相关疾病到related_disease.csv
        pd.DataFrame(list(self.related_disease), columns=["related_disease"]).to_csv(self.related_disease_file, index=False, encoding='utf-8-sig')

if __name__ == "__main__":
    file_path = "./DATA/"
    processor = csv_processor(file_path, API_KEY)
    processor.process_all()