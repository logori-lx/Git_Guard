import pandas as pd
from zai import ZhipuAiClient
import time
import os
from collections import deque
API_KEY = os.getenv("MEDICAL_RAG")
REWRITTEN_QUERY_CACHE_SIZE = 10
class QueryConstructor:
    def __init__(self, api_key=API_KEY):
        self.client = ZhipuAiClient(api_key=api_key)
        self.__rewritten_query_cache_dict = {}
        self.__rewritten_query_cache_queue = deque()
    def __update_rewritten_query_cache(self, query, rewritten_query):
        if len(self.__rewritten_query_cache_dict) == REWRITTEN_QUERY_CACHE_SIZE:
            key = self.__rewritten_query_cache_queue.popleft()
            del self.__rewritten_query_cache_dict[key]
        self.__rewritten_query_cache_dict[query] = rewritten_query
        self.__rewritten_query_cache_queue.append(query)
            
    def get_query(self, query:str):
        rewritten_query = self.__rewritten_query_cache_dict.get(query)
        if rewritten_query == None:
            rewritten_query = self.__rewritten_query(query)
            self.__update_rewritten_query_cache(query,rewritten_query)
        return rewritten_query
        
            
    def __rewritten_query(self, query:str):
        """
        将用户的口语化健康咨询转化为专业的医疗检索Query
        """
        prompt = f"""
        你是一名专业的医疗术语与临床语言优化助手。你的任务是将用户输入的健康咨询或症状描述，改写为更符合医学规范、适合在专业医疗知识库中检索的形式。

        请遵循以下原则：
        1. **术语标准化**：将口语化的身体部位或症状描述转化为标准的医学术语（例如：将“嗓子疼”改为“咽痛”，将“拉肚子”改为“腹泻”）。
        2. **要素提取**：保留关键的症状、持续时间、诱发因素和伴随症状，去除无关的情绪化表达（如“我很担心”、“救命啊”等）。
        3. **逻辑清晰**：如果包含多个症状，请梳理为连贯的临床表述。
        4. **中立客观**：保持客观的陈述语气，不改变原意，不进行诊断（不要凭空捏造病名，除非用户直接询问特定疾病）。
        5. **严格约束**：输出中不要出现任何解释、前缀或后缀，只返回改写后的一句话。

        原查询：{query}
        """

        try:
            response = self.client.chat.completions.create(
                model="glm-4",  # 或者是您实际使用的其他医疗微调模型
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300, # 医疗描述有时比法律短语稍长，稍微增加token
                temperature=0.1 # 降低随机性，追求精准的术语映射
            )
            rewritten_query = response.choices[0].message.content.strip()
            return rewritten_query
        except Exception as e:
            print(f"医疗查询改写失败，使用原查询: {e}")
            return query
    def extract_category(self, query):
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
        return list_content or ["无"]

        
        
        
        

if __name__ == "__main__":
    file_path = "./DATA/"