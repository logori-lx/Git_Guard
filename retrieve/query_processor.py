import sys
import os

# 获取项目根目录的绝对路径（当前文件在 retrieve 文件夹下，上一级就是根目录）
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# 将根目录加入 Python 系统路径
sys.path.append(root_path)
from data_processing.keywords_generation import keywords_generator

def query_processor(query: str) -> str:
    """
    处理用户查询，生成关键词。
    
    :param query: 用户输入的查询字符串
    :return: 处理后的查询字符串（关键词）
    """
    generator = keywords_generator()
    keywords = generator.generate(query)

    return {
        "keywords": keywords,
        "query": query
    }

if __name__ == "__main__":
    query = '糖尿病的症状有哪些？'
    result = query_processor(query)
    print(result)