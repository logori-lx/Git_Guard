#### 生成答案部分：answer_generation.py

input：
rewritten_query.json：改写后的问题
top5_most_similar_vectors.json：每个问题对应的 Top5 召回文档
output: generated_answers.json：每条样本包含 user_input / response / retrieved_contexts

#### ragas评估部分：ragas_evaluation.py

input: generated_answers.json
output: ragas_eval_input.json, ragas_eval_report.json
读取 generated_answers.json 将标准答案整合进json生成 ragas_eval_input.json
DeepSeek + ragas 评估，输出各项指标到 ragas_eval_report.json
