from zai import ZhipuAiClient
import os
from collections import deque
from typing import List, Dict
API_KEY = os.getenv("MEDICAL_RAG")
REWRITTEN_QUERY_CACHE_SIZE = 10
class QueryConstructor:
    def __init__(self, api_key=API_KEY):
        self.client = ZhipuAiClient(api_key=api_key)
        self._rewritten_query_cache_dict = {}
        self._rewritten_query_cache_queue = deque()
    def _update_rewritten_query_cache(self, query, rewritten_query):
        if len(self._rewritten_query_cache_dict) == REWRITTEN_QUERY_CACHE_SIZE:
            key = self._rewritten_query_cache_queue.popleft()
            del self._rewritten_query_cache_dict[key]
        self._rewritten_query_cache_dict[query] = rewritten_query
        self._rewritten_query_cache_queue.append(query)
        
            
    def _rewritten_query(self, query:str):
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
    def get_query(self, query:str):
        rewritten_query = self._rewritten_query_cache_dict.get(query)
        if rewritten_query == None:
            rewritten_query = self._rewritten_query(query)
            self._update_rewritten_query_cache(query,rewritten_query)
        return rewritten_query
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

    from typing import List, Dict, Tuple

    def _build_medical_qa_prompt(self, question: str, passages: List[Dict]) -> Tuple[str, str]:
        """
        返回 (system_prompt, user_prompt)。
        场景：面向普通用户的医疗咨询。
        数据源：相似的历史医患问答对 [{"ask": "...", "answer": "..."}]。
        """
        def _format_qa_context(passages: List[Dict]) -> str:
            """
            辅助函数：将 [{"ask":..., "answer":...}] 格式化为易读的字符串块
            """
            formatted = []
            for i, p in enumerate(passages, 1):
                # 提取 ask 和 answer，处理可能缺失的字段
                ask_text = p.get('ask', '').strip()
                answer_text = p.get('answer', '').strip()
                
                # 截断过长的文本以节省 token（可选）
                if len(ask_text) > 100: ask_text = ask_text[:100] + "..."
                if len(answer_text) > 300: answer_text = answer_text[:300] + "..."

                entry = (
                    f"[案例ID: {i}]\n"
                    f"  - 患者提问: {ask_text}\n"
                    f"  - 医生回答: {answer_text}\n"
                )
                formatted.append(entry)
            
            return "\n".join(formatted)
        # 1. 格式化上下文：专门处理 ask/answer 结构
        context_block = _format_qa_context(passages) if passages else "(未检索到相似病例或医生回答)"

        # 2. System Prompt：确立人设、安全边界和语言风格
        system_prompt = (
            "你是一名专业、温暖且严谨的智能医疗健康助手。你的任务是根据检索到的【历史医患问答】数据，回答当前用户的咨询。\n"
            "**核心原则**：\n"
            "- **角色定位**：你不是医生，不能下确诊诊断。你的回答仅作为参考信息，不能替代线下就医。\n"
            "- **依据优先**：必须基于检索到的“医生回答”进行归纳总结，不要脱离参考资料随意发挥。\n"
            "- **通俗易懂**：用户是普通人，请将医学术语转化为大白话（例如将“上呼吸道感染”解释为“感冒”）。\n"
            "- **安全红线**：若用户描述涉及急危重症（如胸痛、呼吸困难、剧烈外伤），必须优先建议立即就医，而非仅推荐药物。\n"
            "- **禁止造假**：严禁编造检索资料中不存在的药品名称或治疗方案。\n"
        )

        # 3. User Prompt：指导具体的回答逻辑和引用格式
        user_prompt = (
            f"用户当前咨询：\n{question}\n\n"
            f"检索到的相似医患问答（仅供参考）：\n{context_block}\n\n"
            "请严格按照以下步骤生成回答：\n"
            "1)用通俗的语言直接回答用户的疑问。\n"
            "2) 总结检索到的医生观点，告诉用户“通常情况下”这类问题如何处理。\n"
            "3) 若有相关案例，仅说明其如何支持条文含义；无关案例请忽略。\n"
            "4) 禁止输出与医疗建议无关的内容，禁止补充除提供案例以外的案例\n"
        )

        return system_prompt, user_prompt
    def process_medical_query(self, question: str, 
                              context: List[Dict], 
                              model: str = "glm-4",
                              temperature: float = 1.0,
                              max_tokens: int = 65536) -> str:
        sys_prompt, usr_prompt = self._build_medical_qa_prompt(question, context)
        try:
            resp = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": usr_prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            answer = resp.choices[0].message.content.strip()
        except Exception as e:
            # 降级策略：不阻断整体流程，返回简短错误提示与上下文
            answer = f"抱歉，生成过程中出现异常：{e}。以下为检索到的相关病例供参考。"
        return answer
    

    
    

if __name__ == "__main__":
    file_path = "./DATA/"