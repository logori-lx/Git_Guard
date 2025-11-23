#### baseline生成答案部分：baseline_answer_generation.py

input：
rewritten_query.json：改写后的问题
output:
generated_plain_answers.json 但是baseline是普通大模型给出的答案，无检索上下文参数

#### baseline评估部分：

input:
generated_plain_answers.json
top5_most_similar_vectors.json
output: 
ragas_eval_input_plain.json
ragas_eval_report_plain.json
按 id 对齐，把对应的 Top5 文本作为 baseline 的 retrieved_contexts
（recall和faithfulness评分时需要用到这个参数，无这个参数评分始终为0，无retrieved_context的结果放在test1_ragas_eval_report_plain.json）
从 REFERENCE_LIST 按 id 找到标准答案 reference
组装成 ragas 输入格式 转化成 ragas_eval_input_plain.json
调用 ragas_evaluation.run_ragas(...)，输出 baseline 的 ragas 分数 存在 ragas_eval_report_plain.json

#### 依赖：

ragas_evaluation.py(reference)

