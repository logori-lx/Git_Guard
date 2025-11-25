from __future__ import annotations

"""
ragas_evaluation.py

Medical RAG - 答案评估模块 (DeepSeek + ragas)

整体流程:
    rag_generate_and_ragas_test. 先运行 answer_generation.py，得到 generated_answers.json
    2. 本脚本会:
        - 读取 generated_answers.json
        - 为每个问题补充标准答案 reference
        - 生成 ragas_eval_input.json (包含 user_input / response / retrieved_contexts / reference)
        - 调用 ragas + DeepSeek 进行评估，输出各项指标

依赖:
    pip install "ragas>=0.3.0" "datasets>=2.0.0" \
                "langchain>=0.2.0" "langchain-community>=0.2.0"

评估指标:
    - LLMContextRecall      : 上下文召回率
    - Faithfulness          : 忠实度（生成回答是否忠实于检索到的上下文）
    - FactualCorrectness    : 事实正确性（与标准答案的一致程度）
"""

import json
import os
from typing import Any, Dict, List

# ====== rag_generate_and_ragas_test. 配置区域：可以按需修改 ===============================

# 由 answer_generation.py 生成的文件
GENERATED_ANSWERS_PATH = "generated_answers.json"

# 本脚本生成的两个文件
RAGAS_INPUT_PATH = "ragas_eval_input.json"
RAGAS_REPORT_PATH = "ragas_eval_report.json"

# DeepSeek API Key（也可以通过环境变量 DEEPSEEK_API_KEY 提供）
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY") or "sk-3a59029618234b1496de29504891bf78"

# ===============================================================

# ragas + langchain 相关依赖
try:
    from ragas import evaluate  # type: ignore
    from ragas.llms import LangchainLLMWrapper  # type: ignore
    from ragas.metrics import (  # type: ignore
        LLMContextRecall,
        Faithfulness,
        FactualCorrectness,
    )
    from datasets import Dataset  # type: ignore
except ImportError:  # pragma: no cover
    evaluate = None  # type: ignore
    LangchainLLMWrapper = None  # type: ignore
    LLMContextRecall = None  # type: ignore
    Faithfulness = None  # type: ignore
    FactualCorrectness = None  # type: ignore
    Dataset = None  # type: ignore

try:
    from langchain.chat_models import init_chat_model  # type: ignore
except ImportError:  # pragma: no cover
    init_chat_model = None  # type: ignore


# ---------------------------------------------------------------------
# 2. 样例问题的标准答案（reference）
# ---------------------------------------------------------------------
# 说明：
#   - 这里的 18 个问题，已经和你当前的 rewritten_query.json 一一对应
#   - build_ragas_input 会按顺序把 generated_answers.json 的第 i 条
#     和 REFERENCE_LIST[i] 对齐，形成带 reference 的评估样本
# ---------------------------------------------------------------------

REFERENCE_LIST: List[Dict[str, str]] = [
    {
        "user_input": "我有高血压好几年了，一直吃氨氯地平，最近总觉得乏力、偶尔头晕，是药的副作用吗，需要换药吗？",
        "reference": (
            "高血压患者在服用降压药期间出现乏力、轻度头晕，可能与多种因素有关，包括血压控制过低或波动较大、"
            "睡眠不足、贫血、电解质紊乱等，也有可能与药物本身的副作用相关，但不能单纯归咎于某一种药。\n"
            "rag_generate_and_ragas_test）首先建议在一段时间内规律监测血压（包括早晚及出现不适时），记录具体数值，看是否存在血压过低"
            "（比如舒张压明显低于 60mmHg）或波动较大的情况。\n"
            "2）如乏力、头晕症状多在起床、突然站立或劳累之后出现，要警惕体位性低血压或血压过度下降，可先避免猛然起身，"
            "适当放慢动作，并注意补充水分。\n"
            "3）如果近期生活节奏变化较大（比如熬夜、压力增大、饮食不规律），也会引起类似不适，需要先从作息和生活方式调整做起。\n"
            "4）是否需要换药或调整剂量，应由心内科或高血压专科医生根据血压记录、体格检查及必要的化验结果综合判断，"
            "不建议自行停药或更换药物，以免造成血压失控。\n"
            "如症状持续明显、伴有胸闷、视物模糊或晕厥倾向，应尽快就医，避免延误潜在的心脑血管风险。"
        ),
    },
    {
        "user_input": "最近经常心慌、胸口闷，有时一阵阵刺痛，医院心电图说没问题，这种情况需要做哪些进一步检查？",
        "reference": (
            "单次静息心电图正常，并不能完全排除心脏疾病，尤其是发作一阵阵、并非持续存在的心律失常或某些类型的心肌缺血。\n"
            "rag_generate_and_ragas_test）如果心慌、胸闷症状反复出现，建议在心内科医生指导下考虑进一步检查，如：\n"
            "   - 动态心电图（24小时或更长）：用于捕捉发作时的心律变化；\n"
            "   - 心脏超声（彩超）：评估心脏结构和收缩功能；\n"
            "   - 必要时做运动负荷试验或相关心肌缺血评估。\n"
            "2）如伴有胸痛时间较长、活动后症状加重、出冷汗、放射至左肩或下颌等，应警惕冠心病或急性心梗风险，及时急诊就医，而不是只依赖一次普通心电图。\n"
            "3）如果检查提示心脏结构和节律基本正常，同时症状与情绪紧张、咖啡因摄入、睡眠不足等明显相关，也需考虑焦虑、植物神经功能紊乱等非器质性原因，"
            "可在医生指导下进行调整和必要的心理干预。\n"
            "总之，反复心慌、胸闷不建议仅凭一次“心电图没问题”就完全放心，仍应结合病史和体征由心内科医生综合评估。"
        ),
    },
    {
        "user_input": "我是2型糖尿病，空腹血糖还可以，但是餐后经常到12左右，已经注意饮食了，还能从哪些方面调整？",
        "reference": (
            "2 型糖尿病中，单纯空腹血糖控制尚可而餐后血糖偏高的情况比较常见，提示主要问题在餐后血糖峰值和谷值控制。\n"
            "rag_generate_and_ragas_test）饮食层面：除了控制总量，还要关注“吃什么、怎么吃”：\n"
            "   - 适当减少精制碳水（白米饭、甜点、含糖饮料），增加粗粮、蔬菜和优质蛋白；\n"
            "   - 尽量做到细嚼慢咽，避免一次进食过快过多；\n"
            "   - 可尝试少量多餐或减少晚餐主食量，具体需结合体重和营养状况。\n"
            "2）运动方面：在医生允许的前提下，适度安排餐后 30 分钟左右的散步或轻至中等强度活动，有助于降低餐后血糖峰值。\n"
            "3）用药调整：有些降糖药物对空腹血糖效果好，对餐后血糖作用有限；如餐后血糖持续偏高，可在内分泌专科医生指导下评估是否需要调整药物种类、"
            "服用时间或剂量，比如增加针对餐后血糖的药物等。\n"
            "4）血糖监测：建议短期内加强自我监测，记录空腹、餐后 2 小时血糖及饮食内容，带着记录就诊，有利于医生更精准地调整方案。\n"
            "任何用药和剂量改变都应遵循医生建议，避免自行加减药，以免导致低血糖或控制不佳。"
        ),
    },
    {
        "user_input": "这段时间总是晚上难入睡、容易醒、多梦，白天没精神，这种情况算失眠吗，需要吃安眠药吗？",
        "reference": (
            "长期出现入睡困难、夜间易醒、多梦，且白天感到明显乏力、注意力不集中，一般可以视为失眠问题，需要重视。\n"
            "rag_generate_and_ragas_test）首先建议从睡眠卫生和生活方式入手：\n"
            "   - 规律作息，尽量固定上床和起床时间，避免频繁熬夜；\n"
            "   - 睡前尽量减少咖啡、浓茶和大量饮水，避免重口味晚餐；\n"
            "   - 减少睡前长时间刷手机、追剧等强光刺激和情绪波动。\n"
            "2）可以尝试放松训练，如深呼吸、冥想、轻柔拉伸或泡脚等，帮助身体和大脑进入“睡眠模式”。\n"
            "3）是否需要使用安眠药，应由专业医生评估：\n"
            "   - 对于短期严重失眠，可在医生指导下短期、小剂量使用助眠药；\n"
            "   - 长期、反复失眠更推荐在精神科或睡眠门诊接受系统评估，结合认知行为治疗、心理干预等综合方案，避免长期依赖药物。\n"
            "如失眠同时伴有明显情绪低落、焦虑、心悸等，应更加建议尽早就医，寻求专业帮助。"
        ),
    },
    {
        "user_input": "体检发现甲状腺结节TI-RADS 3级，医生说半年复查，我担心会不会变成癌，要不要现在就做手术？",
        "reference": (
            "TI-RADS 3 级甲状腺结节通常提示为“低度可疑”或多为良性病变，恶性风险比较低，因此常规处理以定期随访为主，而不是立即手术。\n"
            "rag_generate_and_ragas_test）一般情况下，TI-RADS 3 级结节建议按医生建议间隔（如 6～12 个月）进行超声复查，观察结节大小、形态和内部回声是否有明显变化。\n"
            "2）如果复查中发现结节快速增大、出现钙化、形态不规则或可疑淋巴结等高危征象，医生可能会建议进一步穿刺活检或考虑手术。\n"
            "3）是否需要手术，不仅取决于恶性风险，还要看结节是否引起压迫症状（如吞咽困难、呼吸不适）、甲状腺功能异常以及患者的整体健康状况等。\n"
            "4）对于目前无明显高危特征的 TI-RADS 3 级结节，遵从医生建议定期随访是一种安全且常规的做法，不必过度焦虑。"
            "如有疑虑，可以到甲状腺专科门诊再进行一次详细评估。"
        ),
    },
    {
        "user_input": "近半年月经越来越不规律，有时40多天来一次，量也减少，还偶尔潮热出汗，我43岁了，这是不是更年期提前？",
        "reference": (
            "43 岁出现月经周期紊乱、经量改变以及潮热出汗等症状，很符合围绝经期（更年期过渡期）的典型表现，属于女性生理变化的一部分，"
            "并不一定是“异常提前”，但仍需排除其他妇科或内分泌疾病。\n"
            "rag_generate_and_ragas_test）建议到妇科或内分泌科就诊，结合病史、体检以及必要时的激素水平检查（如雌激素、卵泡刺激素 FSH 等）进一步确认。\n"
            "2）围绝经期的管理重点在于缓解症状和预防骨质疏松、心血管疾病等风险，包括：\n"
            "   - 规律作息、适度运动、均衡饮食，补充钙和维生素 D；\n"
            "   - 对于潮热、盗汗、情绪波动明显者，可在医生评估后考虑药物或激素相关治疗。\n"
            "3）如伴有异常子宫出血（经量忽多忽少、经期延长、月经间期出血等），需要排除子宫肌瘤、内膜息肉或内膜增生等器质性疾病。\n"
            "综合评估后由专科医生给出个体化的随访和治疗建议，不建议自行长期服用激素或保健品。"
        ),
    },
    {
        "user_input": "最近站起来时总头晕、眼前发黑，偶尔耳鸣，血压偏低，这种低血压需要吃药吗？",
        "reference": (
            "由卧位或坐位突然站立时出现短暂头晕、眼前发黑、耳鸣，且血压测量偏低，常见于体位性低血压或低血容量等情况。"
            "多数轻度病例可以通过生活方式调整得到缓解，并不一定需要药物治疗。\n"
            "rag_generate_and_ragas_test）建议起身动作放慢，从卧位先坐起，再缓慢站立，避免突然改变体位；\n"
            "2）适当增加饮水量，在没有心肾功能不全等禁忌的前提下，可适度增加盐分摄入；\n"
            "3）注意保证睡眠、避免久站和过度劳累，加强下肢肌肉锻炼，提高静脉回流；\n"
            "4）如症状频繁或曾出现晕厥、跌倒等，需要尽快就医，检查心电图、心脏超声、血常规等，排除严重心律失常、贫血以及内分泌疾病。\n"
            "是否需要使用药物，需要由医生在明确病因后综合评估，一般不推荐自行使用升压药。"
        ),
    },
    {
        "user_input": "体检发现血脂偏高，医生建议先生活方式干预不急着用药，我具体应该怎么吃、怎么运动？",
        "reference": (
            "对轻中度血脂异常或刚刚超标的人群，生活方式干预往往是首选措施，也是长期管理的基础。\n"
            "rag_generate_and_ragas_test）饮食方面：\n"
            "   - 减少动物油、肥肉、油炸食品和糕点甜品摄入；\n"
            "   - 多吃蔬菜、水果和全谷物，增加膳食纤维；\n"
            "   - 适量选择富含不饱和脂肪酸的食物，如深海鱼、坚果（少盐少油）、橄榄油等；\n"
            "   - 控制总热量，避免暴饮暴食和夜宵。\n"
            "2）运动方面：\n"
            "   - 每周至少 150 分钟中等强度有氧运动，如快走、骑车、游泳等，可分多次完成；\n"
            "   - 如无禁忌，可以适当加入力量训练，帮助控制体重和改善代谢。\n"
            "3）体重管理和其他因素：\n"
            "   - 如存在超重或中心性肥胖，适度减重往往能显著改善血脂；\n"
            "   - 戒烟限酒，避免长期大量饮酒；\n"
            "   - 定期复查血脂，一般在坚持生活方式干预 3 个月左右复查，根据结果由医生评估是否需要加用药物。\n"
            "如本身已有冠心病、糖尿病或中风等高危因素，是否从一开始就用药，需要由心内科或全科医生综合判断。"
        ),
    },
    {
        "user_input": "小腿晚上睡觉经常抽筋，一晚能抽好几次，白天也容易乏力，是缺钙还是血管有问题，要做什么检查？",
        "reference": (
            "夜间小腿抽筋常见原因包括肌肉疲劳、电解质轻度紊乱（如钙、镁、钾不足）、局部血液循环不佳以及神经系统问题等，"
            "并不能简单地认为只是“缺钙”或“血管堵了”。\n"
            "rag_generate_and_ragas_test）生活方式调整：\n"
            "   - 睡前做小腿和足部的拉伸和放松活动，避免长时间保持同一姿势；\n"
            "   - 注意补充水分，避免大量出汗后不补液；\n"
            "   - 可尝试热敷或温水泡脚，改善局部血液循环。\n"
            "2）需要就医时，可以在医生指导下进行：\n"
            "   - 血常规、电解质、血钙、维生素 D 等化验；\n"
            "   - 必要时行下肢血管超声、神经传导检查，以排除下肢血管病变或周围神经病变。\n"
            "3）如果抽筋频率很高、伴随明显乏力、麻木或行走困难，或者有糖尿病、肾病、甲状腺疾病等基础病史，更应尽早就医明确原因，"
            "避免自行长期服用含钙或止痛药物。"
        ),
    },
    {
        "user_input": "小孩7岁，经常扁桃体发炎，一年要发作好几次，医生建议考虑手术切除，这个手术有必要吗？",
        "reference": (
            "7 岁儿童反复扁桃体炎比较常见，但是否需要手术切除要结合发作频率、严重程度以及是否影响生长发育等综合判断。\n"
            "rag_generate_and_ragas_test）一般情况下，如果：\n"
            "   - 过去 rag_generate_and_ragas_test 年内扁桃体急性发作 ≥7 次，或连续 2 年每年 ≥5 次，或连续 3 年每年 ≥3 次，且发作时有明确高热、化脓、咽痛等表现并有病历记录；\n"
            "   - 或扁桃体明显肥大，导致打鼾、睡眠呼吸暂停、张口呼吸等，影响睡眠和面容发育；\n"
            "   才更倾向于在耳鼻喉科医生评估后考虑手术治疗。\n"
            "2）扁桃体切除是比较成熟的常规手术，但仍存在麻醉风险、术后出血等并发症，需要在正规医院、具备经验的团队中进行。\n"
            "3）如发作次数较少、症状较轻且对药物治疗反应良好，可以继续观察，注意提高体质和避免交叉感染，不必急于手术。\n"
            "建议与耳鼻喉科医生充分沟通，综合孩子的发作频率、严重程度以及对生活学习的影响后再决定是否手术。"
        ),
    },
    {
        "user_input": "最近总感觉心情低落，对什么都提不起兴趣，还经常失眠、食欲差，这样算不算抑郁，要不要去精神科？",
        "reference": (
            "持续的心情低落、兴趣减退、伴随睡眠和食欲明显变化，是抑郁状态的常见表现之一，但是否达到抑郁症的诊断标准，需要由精神科医生通过系统评估来判定。\n"
            "rag_generate_and_ragas_test）如果这种状态已经持续数周甚至更久，并明显影响学习、工作或人际关系，建议尽早到精神科或身心医学科就诊，而不是自己硬扛。\n"
            "2）医生会通过问诊、量表评估和必要检查，综合判断是抑郁症、适应障碍还是其他情绪问题，并制定相应的治疗策略，包括心理治疗、药物治疗或二者结合。\n"
            "3）在等待就诊或配合治疗期间，可以尽量保持规律作息、适度运动、与家人或朋友保持沟通，避免长期独处和过度反刍负面事件。\n"
            "4）如出现明显的绝望感、自伤或自杀想法，应立即向家人、朋友或医疗机构寻求紧急帮助。"
        ),
    },
    {
        "user_input": "大便习惯最近变化比较大，有时候便秘、有时候稀便，颜色也偏深，家里有肠癌病史，这种情况需要马上做肠镜吗？",
        "reference": (
            "大便习惯近期发生明显改变（如便秘与稀便交替、便形变细、大便颜色异常等），尤其合并结直肠癌家族史时，确实需要提高警惕。\n"
            "rag_generate_and_ragas_test）虽然上述症状也可能与功能性肠病、饮食结构变化或感染有关，但在高危人群中，肠镜检查是排查结直肠肿瘤和息肉的关键手段。\n"
            "2）如果年龄已达常规筛查范围（例如 ≥40～45 岁，具体因地区和指南而异），或出现下列警示症状之一，更应尽快就诊：\n"
            "   - 持续大便习惯改变；\n"
            "   - 大便隐血阳性或肉眼可见血便；\n"
            "   - 不明原因的体重下降、贫血、乏力。\n"
            "3）建议到消化科或肛肠科就诊，由医生根据你的年龄、家族史和症状综合评估是否“需要马上做肠镜”，并安排相应的检查计划。\n"
            "切勿因害怕检查而长期拖延，以免错过早期诊断和治疗的机会。"
        ),
    },
    {
        "user_input": "最近咳嗽有黄痰，偶尔痰里有一点血丝，人没发烧，就是胸口有点闷，这种情况严重吗？",
        "reference": (
            "咳嗽有黄痰提示可能存在气道感染或炎症，痰中偶尔带少量血丝，多数情况下与剧烈咳嗽导致小血管破裂有关，但也不能完全排除更严重的肺部疾病。\n"
            "rag_generate_and_ragas_test）如果咳嗽时间较短（如在 2～3 周以内），且伴随感冒或上呼吸道感染病史，多数属于急性气管炎或支气管炎，可在医生指导下对症治疗并观察。\n"
            "2）若咳嗽持续时间较长（超过 3～4 周）、痰中反复或逐渐增多血丝、伴有胸痛、明显气促、夜间盗汗或不明原因消瘦等，应尽快到呼吸科就诊，"
            "完善胸片或肺部 CT 等检查，排除结核、肿瘤等严重疾病。\n"
            "3）任何持续性或加重的咯血都不应掉以轻心，尤其是长期吸烟、既往有肺部疾病或结核病史的人群，更需要规范检查。\n"
            "在明确诊断之前，不建议自行长期使用止咳药或抗生素，以免掩盖病情或导致耐药。"
        ),
    },
    {
        "user_input": "我有慢性胃炎和轻度糜烂，最近总反酸烧心，吃奥美拉唑好一点，但停药又复发，需要长期吃吗？",
        "reference": (
            "慢性胃炎合并反酸、烧心，多数与胃食管反流或胃酸分泌过多有关，质子泵抑制剂（如奥美拉唑）可以有效缓解症状，但是否长期使用要视病情而定。\n"
            "rag_generate_and_ragas_test）首先需要配合饮食和生活方式调整：\n"
            "   - 避免过饱、晚餐过晚、躺着或弯腰前吃大量食物；\n"
            "   - 减少油腻、辛辣、咖啡、浓茶、碳酸饮料和酒精摄入；\n"
            "   - 有条件时可适当抬高床头，避免餐后立即平卧。\n"
            "2）药物方面：\n"
            "   - 一般建议在消化科医生指导下按疗程规范用药，如连续使用 4～8 周后评估症状和内镜复查情况；\n"
            "   - 部分患者可能需要维持治疗或按需间断用药，而不是自行无限期每日服用。长期大剂量使用需权衡利弊并定期随访。\n"
            "3）如症状严重、反复发作或伴有黑便、呕血、持续消瘦等警示信号，应尽快复查胃镜，排除更严重的病变。\n"
            "总之，药物只是治疗的一部分，长期管理还要依赖生活方式的配合以及定期随访，由消化科医生根据具体情况决定是否需要长期或维持用药。"
        ),
    },
    {
        "user_input": "最近手指关节早上起来会僵硬、胀痛，活动一会儿才好，家里有风湿病史，会不会是类风湿关节炎？",
        "reference": (
            "清晨起床时手指关节僵硬、胀痛，活动后逐渐缓解，尤其在有风湿病家族史的情况下，确实需要警惕类风湿关节炎等风湿免疫性疾病的可能。\n"
            "rag_generate_and_ragas_test）类风湿关节炎典型表现为对称性小关节受累（如双手掌指关节、近端指间关节等），晨僵时间可以持续 rag_generate_and_ragas_test 小时以上，随着病程进展可出现关节肿胀、变形甚至功能受限。\n"
            "2）建议尽早就诊风湿免疫科，完善相关检查，包括关节 X 线或超声、类风湿因子、抗 CCP 抗体以及炎症指标（ESR、CRP）等，以便尽早明确诊断。\n"
            "3）类风湿关节炎若能在早期得到规范治疗，可以明显减缓关节破坏和残疾风险，因此“早发现、早诊断、早治疗”非常关键。\n"
            "4）在未明确诊断前，不建议长期自行服用止痛药或激素，以免掩盖病情或带来其他副作用。"
        ),
    },
    {
        "user_input": "之前得过轻微脑梗，现在一直吃阿司匹林和他汀，最近感觉记性差、反应慢，是正常老化还是脑供血不好？",
        "reference": (
            "既往有脑梗病史的患者，即使损伤较轻，也可能遗留一定程度的认知功能减退，如记忆力、注意力、反应速度等方面的变化。"
            "随着年龄增长，还会叠加自然老化因素和其他疾病（如高血压、糖尿病等）的影响。\n"
            "rag_generate_and_ragas_test）如果近期记忆力和反应能力较之前明显下降，建议尽快在神经内科门诊复查，包括：\n"
            "   - 影像学检查（如头颅 CT/MRI）了解是否有新的梗塞灶或脑萎缩进展；\n"
            "   - 认知功能量表评估，判断是否存在轻度认知障碍或痴呆的早期表现；\n"
            "   - 同时评估血压、血糖、血脂控制情况。\n"
            "2）阿司匹林和他汀属于常规的卒中二级预防用药，一般应在医生指导下长期规范服用，不要自行停药或减量。\n"
            "3）除了药物治疗，康复训练、认知训练以及控制危险因素（戒烟限酒、控制体重、改善睡眠）也对大脑功能维护非常重要。\n"
            "是否属于“正常老化”或“脑供血不足”，需要通过上述检查和医生综合判断，不能仅凭主观感觉判断。"
        ),
    },
    {
        "user_input": "一直有过敏性鼻炎，打喷嚏流鼻涕反反复复，用激素类喷鼻剂有效但停药又复发，这种喷剂可以长期用吗？",
        "reference": (
            "鼻用糖皮质激素是过敏性鼻炎的一线标准治疗之一，局部喷雾形式全身吸收较少，相对于口服激素安全性更高，在正确使用和定期复查的前提下，"
            "可以中长期甚至长期维持治疗。\n"
            "rag_generate_and_ragas_test）关键在于“用法和监测”：\n"
            "   - 严格按照说明或医生指导的剂量和频率使用，不要自行大幅增加用量；\n"
            "   - 喷药时注意避开鼻中隔，稍微向外侧壁方向喷，以减少局部不良反应。\n"
            "2）多数指南建议，在症状控制稳定一段时间后，可以尝试逐渐减量或间断使用，而不是一直保持最大剂量；具体方案需由耳鼻喉科或变态反应科医生制定。\n"
            "3）如使用过程中出现频繁鼻出血、明显干燥、刺激感或其他不适，应及时复诊，调整用药或排除其他鼻腔疾病。\n"
            "4）同时配合避免过敏原、鼻腔冲洗和生活方式调整，可以减少疾病复发和对药物的依赖程度。\n"
            "总之，激素类喷鼻剂在规范使用的前提下，是可以长期使用的，但需要在专科医生指导下定期评估和调整。"
        ),
    },
    {
        "user_input": "体检发现心电图“窦性心律不齐”，平时偶尔心慌，这个严重吗，需要特别治疗吗？",
        "reference": (
            "窦性心律不齐，尤其是与呼吸相关的窦性心律不齐，在青少年和部分成年人中很常见，大多属于生理性现象，一般不影响寿命和生活质量。\n"
            "rag_generate_and_ragas_test）如果只是体检心电图提示“窦性心律不齐”，而没有明显的晕厥、严重心悸、胸痛或活动耐量骤降等症状，多数情况下无需特别治疗，只需定期随访观察。\n"
            "2）平时偶尔短暂心慌，也可能与情绪紧张、咖啡因摄入、熬夜等有关，可先从生活方式调整入手：保证睡眠、减压、减少浓茶咖啡等刺激性饮品。\n"
            "3）如心慌明显、持续时间较长或伴有胸痛、头晕、黑蒙等，应在心内科医生指导下进一步检查（如动态心电图等），排除其他类型的心律失常。\n"
            "4）在明确为良性的窦性心律不齐并症状轻微的前提下，一般不需要特别药物治疗，只需要注意生活习惯和定期复查即可。\n"
            "如有担心，可携带既往检查结果就诊心内科，由医生结合你的整体情况给予更有针对性的解释和建议。"
        ),
    },
]


# ---------------------------------------------------------------------
# 3. I/O 工具函数
# ---------------------------------------------------------------------

def load_generated_answers(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"{path} 顶层 JSON 必须是列表(list)。")

    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"{path} 中第 {idx} 条记录不是对象(dict)。")
        for key in ("user_input", "response", "retrieved_contexts"):
            if key not in item:
                raise ValueError(f"{path} 中第 {idx} 条记录缺少字段: {key}")

    return data


def build_ragas_input(
    generated: List[Dict[str, Any]],
    references: List[Dict[str, str]],
) -> List[Dict[str, Any]]:
    """
    将 generated_answers.json 与 REFERENCE_LIST 对齐，生成 ragas_eval_input.json 所需结构。

    对齐策略：
        - 默认按「顺序」一一对应：第 i 条 generated 对应第 i 条 reference
        - 如果两者长度不一致，则取较小值并给出提示
    """
    n = min(len(generated), len(references))
    if len(generated) != len(references):
        print(
            f"[RAGAS] 警告: generated_answers.json 有 {len(generated)} 条，"
            f"REFERENCE_LIST 有 {len(references)} 条，将按前 {n} 条对齐。"
        )

    ragas_samples: List[Dict[str, Any]] = []
    for i in range(n):
        g = generated[i]
        r = references[i]

        ragas_samples.append(
            {
                "user_input": g["user_input"],
                "response": g["response"],
                "retrieved_contexts": g.get("retrieved_contexts", []),
                "reference": r["reference"],
            }
        )

    return ragas_samples


def save_json(data: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------
# 4. 调用 ragas + DeepSeek 进行评估
# ---------------------------------------------------------------------

def run_ragas(
    ragas_samples: List[Dict[str, Any]],
    deepseek_api_key: str,
) -> Dict[str, float]:
    if evaluate is None or LangchainLLMWrapper is None or Dataset is None:
        raise ImportError(
            "ragas 或 datasets 未正确安装。请先执行：\n"
            "  pip install 'ragas>=0.3.0' 'datasets>=2.0.0'"
        )
    if LLMContextRecall is None or Faithfulness is None or FactualCorrectness is None:
        raise ImportError(
            "ragas.metrics 未正确导入，请检查 ragas 版本。"
        )
    if init_chat_model is None:
        raise ImportError(
            "langchain 未安装，请先执行：\n"
            "  pip install 'langchain>=0.2.0' 'langchain-community>=0.2.0'"
        )

    if not deepseek_api_key or "在这里填你的DeepSeek_APIKey" in deepseek_api_key:
        raise ValueError(
            "未提供 DeepSeek API Key。请在文件顶部 DEEPSEEK_API_KEY 位置填入你的 Key，"
            "或者通过环境变量 DEEPSEEK_API_KEY 提供。"
        )

    if not ragas_samples:
        raise ValueError("ragas_samples 为空，无法进行评估。")

    # rag_generate_and_ragas_test. 构造 HuggingFace Dataset
    dataset = Dataset.from_list(ragas_samples)  # type: ignore[arg-type]

    # 2. 初始化 DeepSeek 作为评估 LLM
    llm = init_chat_model(
        model="deepseek-chat",
        api_key=deepseek_api_key,
        api_base="https://api.deepseek.com/",
        temperature=0,
        model_provider="deepseek",
    )
    evaluator_llm = LangchainLLMWrapper(llm)

    # 3. 调用 ragas.evaluate
    result = evaluate(
        dataset=dataset,
        metrics=[
            LLMContextRecall(),
            Faithfulness(),
            FactualCorrectness(),
        ],
        llm=evaluator_llm,
        column_map={
            "question": "user_input",
            "answer": "response",
            "contexts": "retrieved_contexts",
            "ground_truth": "reference",
        },
    )

    # 4. 将结果转成简单的 {metric: score} 字典
    metrics_dict: Dict[str, float] = {}

    try:
        if hasattr(result, "to_pandas"):
            df = result.to_pandas()  # type: ignore[attr-defined]
            for col in df.columns:
                try:
                    metrics_dict[col] = float(df[col].mean())
                except Exception:
                    continue
        else:
            metrics_dict = dict(result)  # type: ignore[arg-type]
    except Exception:
        try:
            metrics_dict = dict(result)  # type: ignore[arg-type]
        except Exception:
            metrics_dict = {}

    return metrics_dict


# ---------------------------------------------------------------------
# 5. 主入口
# ---------------------------------------------------------------------

def main() -> None:
    print("=== Medical RAG: RAGAS 评估模块 ===\n")

    # rag_generate_and_ragas_test. 读取 generated_answers.json
    try:
        generated = load_generated_answers(GENERATED_ANSWERS_PATH)
    except Exception as e:
        print(f"[RAGAS] 读取 {GENERATED_ANSWERS_PATH} 失败：{e}")
        return

    print(f"[RAGAS] 成功读取 generated_answers.json，共 {len(generated)} 条样本。")

    # 2. 构造 ragas_eval_input.json
    ragas_samples = build_ragas_input(generated, REFERENCE_LIST)
    save_json(ragas_samples, RAGAS_INPUT_PATH)
    print(f"[RAGAS] 已将评估数据写入: {RAGAS_INPUT_PATH}")

    # 3. 调用 ragas + DeepSeek 评估
    try:
        metrics = run_ragas(ragas_samples, deepseek_api_key=DEEPSEEK_API_KEY)
    except ImportError as e:
        print("[RAGAS] 依赖未安装，无法执行评估。")
        print("错误信息：", e)
        return
    except Exception as e:
        print("[RAGAS] 调用 ragas 评估时出错：", e)
        print("如果暂时只需要生成 ragas_eval_input.json，可以忽略此错误。")
        return

    # 4. 打印并保存评估结果
    if metrics:
        print("\n[RAGAS] 评估结果（各指标平均分）:")
        for name, score in metrics.items():
            try:
                print(f"  {name}: {float(score):.4f}")
            except Exception:
                print(f"  {name}: {score}")
        save_json(metrics, RAGAS_REPORT_PATH)
        print(f"\n[RAGAS] 评估结果已写入: {RAGAS_REPORT_PATH}")
    else:
        print("[RAGAS] 评估返回结果为空，请检查 ragas 版本或输入数据。")


if __name__ == "__main__":
    main()
