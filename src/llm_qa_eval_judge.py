import mysql.connector
import json
import os
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from common import json_tools
from common.system import ErrorCounter
from common.timer import timer
from common.logger import setup_logger, get_logger, TimestampType
from common.config import load_config, get_db_config, get_api_key_secret_key
from common.entity import StandardQA, ModelAnswer, IJudgeResult
from common.llm_api import ModelConfig, load_models_configs
from common.llm_api import ILLM, MockLLM, LLM
from common.api_key_encryptor import ApiKeyEncryptor
from validator.judge_result_validators import JudgeResultValidator
import argparse
from dataclasses import dataclass

# 读取配置
CONFIG_NAME = "llm_qa_eval_judge"
CONFIG = load_config()
DB_CONFIG = get_db_config(CONFIG)
LOG_FILE = CONFIG[CONFIG_NAME]['log_file']
# 如果一个裁判大模型API出错次数超过MAX_API_ERROR_LIMIT次, 则停止使用该裁判大模型进行评估
MAX_API_ERROR_LIMIT = 3

if __name__ == "__main__":
    # 将在main函数中配置
    logger = None  
else:
    # 当作为模块导入时，使用默认配置
    logger = get_logger(__name__)

# 裁判模型的系统提示
JUDGE_SYSTEM_PROMPT = """
You need to evaluate the quality of a candidate answer to a Linux kernel-related question. Provided information includes:
- question: The question being asked
- reference_answer: The correct answer
- key_points: Key knowledge points from the reference answer (numbered)
- candidate_answer: The answer to be evaluated

### Your Task

Analyze the candidate answer and output a strictly formatted JSON evaluation result containing the following fields:

- `"key_points_evaluation"`: An object containing:
  - `"missed"`: Key point numbers absent from the answer
  - `"partial"`: Key point numbers mentioned but lacking details
  - `"matched"`: Key point numbers fully covered

For each of the following fields, you MUST use exact, unmodified text from the candidate answer:
- `"factual_errors"`: List of factually incorrect statements
- `"vague_statements"`: List of ambiguous statements
- `"partially_correct_but_misleading"`: List of misleading statements
- `"irrelevant_but_correct"`: List of irrelevant statements

Each item in these lists must contain:
- `"exact_text"`: The exact text copied from the answer (DO NOT modify or summarize)
- `"explanation"`: Brief explanation of the issue

### Important Rules: Reference Original Text

1. Reference text must be exact:
   - Text must be copied directly from the answer, without modification
   - No summarizing, paraphrasing, or adding/removing words
   - No changing punctuation or formatting
   - If text spans multiple sentences, include complete sentences

2. Reference must be continuous:
   - Text must be a continuous segment from the answer
   - If multiple parts need to be referenced, create separate entries
   - Do not combine or concatenate text from different parts of the answer

### Example Input

```json
{
  "question": "How does the Linux kernel handle process scheduling?",
  "reference_answer": "The Linux kernel uses the Completely Fair Scheduler (CFS) as the default scheduler. CFS manages processes using a red-black tree (RB-Tree) and determines process priority based on vruntime. Additionally, the Linux kernel provides real-time scheduling policies such as SCHED_FIFO and SCHED_RR to meet different scheduling needs.",
  "key_points": {
    "1": "The Linux kernel uses CFS as the default scheduler",
    "2": "CFS manages processes using a red-black tree (RB-Tree)",
    "3": "CFS determines process priority based on vruntime",
    "4": "The Linux kernel provides real-time scheduling policies such as SCHED_FIFO and SCHED_RR"
  },
  "candidate_answer": "Linux uses Completely Fair Scheduler(CFS) for scheduling, and CFS allocates CPU time using time slices. Linux employs a certain data structure to manage processes. Additionally, Linux supports BPF for flexible scheduling."
}
```

### Example Output

Only output the JSON object, no other text or comments.

```json
{
  "key_points_evaluation": {
    "missed": [3, 4],
    "partial": [2],
    "matched": [1]
  },
  "factual_errors": [
    {
      "exact_text": "CFS allocates CPU time using time slices",
      "explanation": "CFS does not use time slices, it uses vruntime for fair scheduling"
    }
  ],
  "vague_statements": [
    {
      "exact_text": "Linux employs a certain data structure to manage processes",
      "explanation": "The statement is vague as it does not specify which data structure (red-black tree) is used"
    }
  ],
  "partially_correct_but_misleading": [],
  "irrelevant_but_correct": [
    {
      "exact_text": "Linux supports BPF for flexible scheduling",
      "explanation": "While BPF can be used for scheduling, it is not directly related to the core scheduling mechanisms being discussed"
    }
  ]
}
```
"""

# 模拟裁判模型的推理
MOCK_JUDGE_REASONING:str = """Mock Judge Reasoning"""

# 模拟裁判模型的输出
MOCK_JUDGE_RESULT:str = \
"""
{
  "key_points_evaluation": {
    "missed": [3, 4],
    "partial": [2],
    "matched": [1]
  },
  "factual_errors": [
    {
      "exact_text": "CFS allocates CPU time using time slices",
      "explanation": "[Mock] CFS does not use time slices, it uses vruntime for fair scheduling"
    }
  ],
  "vague_statements": [
    {
      "exact_text": "Linux employs a certain data structure to manage processes",
      "explanation": "[Mock] The statement is vague as it does not specify which data structure (red-black tree) is used"
    }
  ],
  "partially_correct_but_misleading": [],
  "irrelevant_but_correct": [
    {
      "exact_text": "Linux supports BPF for flexible scheduling",
      "explanation": "[Mock] While BPF can be used for scheduling, it is not directly related to the core scheduling mechanisms being discussed"
    }
  ]
}"""

class JudgeResult(IJudgeResult):
    """裁判模型评估结果的封装类"""
    def __init__(self, model_answer_id:int, judge_model_id:int, reasoning_content:str, content:str):
        """初始化评估结果。
        
        Args:
            model_answer_id: 模型回答ID
            judge_model_id: 裁判模型ID
            reasoning_content: 推理内容
            content: 评估结果内容，JSON格式
        """
        # 初始化
        self.model_answer_id = model_answer_id
        self.judge_model_id = judge_model_id
        # 评估结果，均初始化为None
        # 维度1：关键点
        self.missed_key_points = None
        self.partial_key_points = None
        self.matched_key_points = None
        # 维度2：事实正确性
        self.factual_errors = None
        self.partially_correct_but_misleading = None
        # 维度3：表达清晰度
        self.vague_statements = None
        self.irrelevant_but_correct = None
        # 裁判模型的原始输出
        self.reasoning_content = reasoning_content
        self.content = content
        # 裁判模型的输出是否出错
        self.is_eval_error = False

        # 使用JudgeResultValidator解析和验证裁判模型的原始输出
        parse_success, evaluation_data, parse_error = JudgeResultValidator.parse_judge_result_json(content)
        if parse_success:
            # 验证结构
            is_valid, validation_errors, _ = JudgeResultValidator.validate_judge_result_structure(evaluation_data)
            if is_valid:
                # 解析成功且结构正确，提取数据
                key_points_eval = evaluation_data['key_points_evaluation']
                self.missed_key_points = key_points_eval['missed']
                self.partial_key_points = key_points_eval['partial']
                self.matched_key_points = key_points_eval['matched']
                self.factual_errors = evaluation_data['factual_errors']
                self.partially_correct_but_misleading = evaluation_data['partially_correct_but_misleading']
                self.vague_statements = evaluation_data['vague_statements']
                self.irrelevant_but_correct = evaluation_data['irrelevant_but_correct']
            else:
                # 结构验证失败
                self.is_eval_error = True
                logger.error(f"裁判模型评估结果结构验证失败: {validation_errors}")
        else:
            # JSON解析失败
            self.is_eval_error = True
            logger.error(f"裁判模型评估结果解析错误: {parse_error}")

    def get_model_answer_id(self):
        return self.model_answer_id
    def get_judge_model_id(self):
        return self.judge_model_id

    def get_missed_key_points(self):
        return self.missed_key_points
    def get_partial_key_points(self):
        return self.partial_key_points
    def get_matched_key_points(self):
        return self.matched_key_points
    
    def get_factual_errors(self):
        return self.factual_errors    
    def get_partially_correct_but_misleading(self):
        return self.partially_correct_but_misleading
    
    def get_vague_statements(self):
        return self.vague_statements
    def get_irrelevant_but_correct(self):
        return self.irrelevant_but_correct
    
    def get_reasoning_content(self):
        return self.reasoning_content
    def get_content(self):
        return self.content
    def get_is_eval_error(self):
        return self.is_eval_error

    def format_answer_raw_info(self):
        """"格式化模型的返回"""
        reasoning_part = "无模型思考过程" if not self.reasoning_content else f"【模型思考过程】\n{self.reasoning_content}"
        answer_part = "无模型回答" if not self.content else f"【模型回答】\n{self.content}"
        return f"{reasoning_part}\n{answer_part}"

    def __str__(self):
        return json.dumps(self.__dict__, indent=2, ensure_ascii=False)
    
    def __repr__(self):
        return self.__str__()

def get_user_content_of_judge_prompt(question, modelAnswer:ModelAnswer, correct_answer, key_points:List):
    """获取用于裁判模型评估的用户提示内容。
    
    Args:
        question: 问题内容
        modelAnswer: 模型回答对象
        correct_answer: 正确答案
        key_points: 关键知识点列表
        
    Returns:
        str: JSON格式的评估提示内容
    """
    # 将key_points列表转换为字典格式
    key_points_dict = {str(i+1): point for i, point in enumerate(key_points)}
    
    prompt_data = {
        "question": question,
        "reference_answer": correct_answer,
        "key_points": key_points_dict,
        "candidate_answer": modelAnswer.get_answer()
    }
    return json.dumps(prompt_data, indent=2, ensure_ascii=False)

def get_unjudged_standard_qa_batch(limit, llm_model_id, judge_llm_model_ids) -> List[StandardQA]:
    """获取指定模型的问答对中至少有一个裁判模型未评估过的标准问答对批次。
    
    Args:
        limit: 限制返回数量
        llm_model_id: 被测模型的数据库ID
        judge_llm_model_ids: 裁判模型的数据库ID列表
        
    Returns:
        List[StandardQA]: 标准问答对列表
    """
    if not judge_llm_model_ids:
        logger.warning("没有提供裁判模型ID列表，将返回空列表")
        return []
        
    conn = mysql.connector.connect(**DB_CONFIG)
    try:
        with conn.cursor(dictionary=True) as cursor:
            # 创建用于查询的占位符 (%s) 
            placeholders = ', '.join(['%s'] * len(judge_llm_model_ids))
            query = f"""
                WITH model_answers AS (
                    SELECT DISTINCT mea.id as model_answer_id, mea.qa_id
                    FROM model_eval_answers mea
                    WHERE mea.model_id = %s
                ),
                judged_answers AS (
                    SELECT ma.qa_id, COUNT(DISTINCT mer.judge_model_id) as judged_count
                    FROM model_answers ma
                    LEFT JOIN model_eval_results mer ON ma.model_answer_id = mer.model_answer_id
                    AND mer.judge_model_id IN ({placeholders})
                    GROUP BY ma.qa_id
                )
                SELECT DISTINCT r.id                 AS id,
                                q.source_question_id AS source_question_id,
                                a.source_answer_id   AS source_answer_id,
                                q.question           AS question,
                                a.answer             AS answer,
                                a.key_points         AS key_points,
                                q.status_id          AS q_status_id,
                                q.status_reason      AS q_status_reason,
                                a.status_id          AS a_status_id,
                                a.status_reason      AS a_status_reason
                FROM standard_qa_relations r
                        JOIN standard_questions q ON r.question_id = q.id
                        JOIN standard_answers a ON r.answer_id = a.id
                        JOIN model_answers ma ON r.id = ma.qa_id
                        JOIN judged_answers ja ON r.id = ja.qa_id
                WHERE ja.judged_count < %s
                AND ((q.status_id = 1 or q.status_id = 2 or q.status_id = 4) AND
                    (a.status_id = 1 or a.status_id = 2 or a.status_id = 4))
                LIMIT %s
            """
            # 执行查询时将参数传递给 execute
            cursor.execute(query, (llm_model_id,) + tuple(judge_llm_model_ids) + (len(judge_llm_model_ids), limit))
            return [StandardQA.from_db_row(row) for row in cursor.fetchall()]
    finally:
        conn.close()

def get_unjudged_standard_qa(qa_ids, limit, llm_model_id, judge_llm_model_ids) -> List[StandardQA]:
    """根据问答对ID列表获取指定模型中至少有一个裁判模型未评估过的标准问答对。
    
    Args:
        qa_ids: 问答对ID列表
        limit: 限制返回数量
        llm_model_id: 被测模型的数据库ID
        judge_llm_model_ids: 裁判模型的数据库ID列表
        
    Returns:
        List[StandardQA]: 标准问答对列表
    """
    if not judge_llm_model_ids:
        logger.warning("没有提供裁判模型ID列表，将返回空列表")
        return []
        
    conn = mysql.connector.connect(**DB_CONFIG)
    try:
        with conn.cursor(dictionary=True) as cursor:
            # 创建用于查询的占位符 (%s) 
            qa_placeholders = ', '.join(['%s'] * len(qa_ids))
            judge_placeholders = ', '.join(['%s'] * len(judge_llm_model_ids))
            query = f"""
                WITH model_answers AS (
                    SELECT DISTINCT mea.id as model_answer_id, mea.qa_id
                    FROM model_eval_answers mea
                    WHERE mea.model_id = %s
                    AND mea.qa_id IN ({qa_placeholders})
                ),
                judged_answers AS (
                    SELECT ma.qa_id, COUNT(DISTINCT mer.judge_model_id) as judged_count
                    FROM model_answers ma
                    LEFT JOIN model_eval_results mer ON ma.model_answer_id = mer.model_answer_id
                    AND mer.judge_model_id IN ({judge_placeholders})
                    GROUP BY ma.qa_id
                )
                SELECT DISTINCT r.id                 AS id,
                                q.source_question_id AS source_question_id,
                                a.source_answer_id   AS source_answer_id,
                                q.question           AS question,
                                a.answer             AS answer,
                                a.key_points         AS key_points,
                                q.status_id          AS q_status_id,
                                q.status_reason      AS q_status_reason,
                                a.status_id          AS a_status_id,
                                a.status_reason      AS a_status_reason
                FROM standard_qa_relations r
                        JOIN standard_questions q ON r.question_id = q.id
                        JOIN standard_answers a ON r.answer_id = a.id
                        JOIN model_answers ma ON r.id = ma.qa_id
                        JOIN judged_answers ja ON r.id = ja.qa_id
                WHERE ja.judged_count < %s
                AND ((q.status_id = 1 or q.status_id = 2 or q.status_id = 4) AND
                        (a.status_id = 1 or a.status_id = 2 or a.status_id = 4))
                LIMIT %s
            """
            # 执行查询时将参数传递给 execute
            cursor.execute(query, (llm_model_id,) + tuple(qa_ids) + tuple(judge_llm_model_ids) + (len(judge_llm_model_ids), limit))
            return [StandardQA.from_db_row(row) for row in cursor.fetchall()]
    finally:
        conn.close()

def get_saved_model_answer(qa_id:int, llm_model_id:str) -> ModelAnswer:
    """从数据库获取已保存的模型回答。
    
    Args:
        qa_id: 问答对ID
        llm_model_id: 模型ID
        
    Returns:
        ModelAnswer: 模型回答对象，如果不存在则返回None
    """
    conn = mysql.connector.connect(**DB_CONFIG)
    try:
        logger.info(f"正在从数据库获取已保存的模型回答: qa_id={qa_id}, llm_model_id={llm_model_id}")    
        with conn.cursor(dictionary=True) as cursor:
            cursor.execute("""
                SELECT mea.id           AS id,
                    mea.qa_id        AS qa_id,
                    llm.id           AS llm_model_id,
                    llm.model_name   AS model_name,
                    llm.model_remark AS model_id,
                    mea.answer_content,
                    mea.reasoning_content,
                    mea.updated_at,
                    mea.created_at
                FROM model_eval_answers mea
                        JOIN llm_models llm on mea.model_id = llm.id
                WHERE mea.qa_id = %s AND mea.model_id = %s
            """, (qa_id, llm_model_id))
            result = cursor.fetchone()
            if result:
                return ModelAnswer.from_db_row(result)
            return None
    finally:
        conn.close()


def get_saved_model_answers_batch(qa_id_list: List[int], llm_model_id: str) -> dict[int, ModelAnswer]:
    """批量从数据库获取已保存的模型回答。
    
    Args:
        qa_id_list: 问答对ID列表
        llm_model_id: 模型ID
        
    Returns:
        dict[int, ModelAnswer]: 问答对ID到模型回答对象的映射，不存在的回答不会包含在结果中
    """
    if not qa_id_list:
        return {}
        
    conn = mysql.connector.connect(**DB_CONFIG)
    try:
        logger.info(f"正在批量获取已保存的模型回答: qa_count={len(qa_id_list)}, llm_model_id={llm_model_id}")
        with conn.cursor(dictionary=True) as cursor:
            # 创建占位符
            placeholders = ', '.join(['%s'] * len(qa_id_list))
            query = f"""
                SELECT mea.id           AS id,
                    mea.qa_id        AS qa_id,
                    llm.id           AS llm_model_id,
                    llm.model_name   AS model_name,
                    llm.model_remark AS model_id,
                    mea.answer_content,
                    mea.reasoning_content,
                    mea.updated_at,
                    mea.created_at
                FROM model_eval_answers mea
                        JOIN llm_models llm on mea.model_id = llm.id
                WHERE mea.qa_id IN ({placeholders}) AND mea.model_id = %s
            """
            cursor.execute(query, qa_id_list + [llm_model_id])
            results = cursor.fetchall()
            # 构建结果字典
            answer_dict = {}
            for result in results:
                model_answer = ModelAnswer.from_db_row(result)
                answer_dict[result['qa_id']] = model_answer
            return answer_dict
    finally:
        conn.close()

def judge_model_answer(qa:StandardQA, model_answer:ModelAnswer, judge_llm:ILLM) -> JudgeResult:
    """使用裁判模型评估模型答案的准确性。
    
    Args:
        qa: 标准问答对
        model_answer: 被测模型的回答
        judge_llm: 裁判模型客户端
        
    Returns:
        JudgeResult: 裁判结果对象，评估失败时返回None
    """
    try:
        judge_model_config = judge_llm.get_model_config()
        logger.info(f"使用裁判模型评估答案的准确性: {judge_model_config.format_single_info()}, qa_id={qa.id}, model_answer_id={model_answer.id}")
        user_content_for_judge = get_user_content_of_judge_prompt(
            qa.split_question, 
            model_answer, 
            qa.split_answer, 
            qa.key_points
        )
        logger.info(f"User content for judge:\n{user_content_for_judge}")
        reasoning_content, answer_content = judge_llm.autoCall(
            system_role=JUDGE_SYSTEM_PROMPT,
            user_content=user_content_for_judge,
            temperature=0.2,
            max_tokens=4096,
        )
        judgeResult = JudgeResult(
                model_answer_id=model_answer.id,
                judge_model_id=judge_model_config.id,
                content=answer_content, reasoning_content=reasoning_content
            )
        logger.info(f"{judge_model_config.model_name}({judge_model_config.id}) 对候选回答({model_answer.id})的评估结果: \n{judgeResult.format_answer_raw_info()}")
        return judgeResult
    except Exception as e:
        logger.error(f"裁判模型 {judge_llm.get_model_config().model_name}({judge_llm.get_model_config().id}) 评估问答对 {qa.id} 时出错: {e}", exc_info=True)
        return None

def get_unjudged_qa_for_judge(qa_answers: List[Tuple[StandardQA, ModelAnswer]], judge_llm: ILLM) -> List[Tuple[StandardQA, ModelAnswer]]:
    """获取指定裁判模型未评估过的问答对。
    
    Args:
        qa_answers: 问答对和对应的模型回答列表
        judge_llm: 裁判模型
        
    Returns:
        List[Tuple[StandardQA, ModelAnswer]]: 该裁判模型未评估过的问答对列表
    """
    judge_model_id = judge_llm.get_model_config().id
    conn = mysql.connector.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cursor:
            # 获取已经评估过的model_answer_id列表
            model_answer_ids = [model_answer.id for _, model_answer in qa_answers]
            if not model_answer_ids:
                return []
                
            placeholders = ', '.join(['%s'] * len(model_answer_ids))
            cursor.execute(f"""
                SELECT DISTINCT model_answer_id
                FROM model_eval_results
                WHERE model_answer_id IN ({placeholders})
                AND judge_model_id = %s
            """, tuple(model_answer_ids) + (judge_model_id,))
            judged_ids = {row[0] for row in cursor.fetchall()}
            
            # 过滤出未评估过的问答对
            return [(qa, model_answer) for qa, model_answer in qa_answers if model_answer.id not in judged_ids]
    finally:
        conn.close()

@dataclass
class JudgeModelParams:
    """评估模型参数类"""
    model_config: ModelConfig
    judgeLLMs: List[ILLM]
    qa_workers: int
    batch_size: int
    limit_qa_count: int
    qa_src: str
    qa_id_list: List[int]
    api_key_encryptor: ApiKeyEncryptor

def save_single_evaluation_result(judge_result: JudgeResult):
    """保存单个评估结果到数据库。
    
    Args:
        judge_result: 裁判评估结果对象
    """
    conn = mysql.connector.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO model_eval_results
                (model_answer_id, judge_model_id, missed_key_points, partial_key_points,
                matched_key_points, factual_errors, vague_statements,
                partially_correct_but_misleading, irrelevant_but_correct,
                reasoning_content, content, is_eval_error)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                judge_result.get_model_answer_id(),
                judge_result.get_judge_model_id(),
                json.dumps(judge_result.get_missed_key_points()),
                json.dumps(judge_result.get_partial_key_points()),
                json.dumps(judge_result.get_matched_key_points()),
                json.dumps(judge_result.get_factual_errors()),
                json.dumps(judge_result.get_vague_statements()),
                json.dumps(judge_result.get_partially_correct_but_misleading()),
                json.dumps(judge_result.get_irrelevant_but_correct()),
                judge_result.get_reasoning_content(),
                judge_result.get_content(),
                judge_result.get_is_eval_error()
            ))
            conn.commit()
    finally:
        conn.close()
        logger.info(f"已保存裁判模型 {judge_result.get_judge_model_id()} 对回答 {judge_result.get_model_answer_id()} 的评估结果")

def save_evaluation_results_batch(judge_results: List[JudgeResult]):
    """批量保存评估结果到数据库。
    
    Args:
        judge_results: 裁判评估结果列表
    """
    if not judge_results:
        return
    
    conn = mysql.connector.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cursor:
            # 批量插入评估结果
            values = [
                (
                    judge_result.get_model_answer_id(),
                    judge_result.get_judge_model_id(),
                    json.dumps(judge_result.get_missed_key_points()),
                    json.dumps(judge_result.get_partial_key_points()),
                    json.dumps(judge_result.get_matched_key_points()),
                    json.dumps(judge_result.get_factual_errors()),
                    json.dumps(judge_result.get_vague_statements()),
                    json.dumps(judge_result.get_partially_correct_but_misleading()),
                    json.dumps(judge_result.get_irrelevant_but_correct()),
                    judge_result.get_reasoning_content(),
                    judge_result.get_content(),
                    judge_result.get_is_eval_error()
                )
                for judge_result in judge_results
            ]
            
            cursor.executemany("""
                INSERT INTO model_eval_results
                (model_answer_id, judge_model_id, missed_key_points, partial_key_points,
                matched_key_points, factual_errors, vague_statements,
                partially_correct_but_misleading, irrelevant_but_correct,
                reasoning_content, content, is_eval_error)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, values)
            
            conn.commit()
            logger.info(f"批量保存了 {len(judge_results)} 个评估结果")
    finally:
        conn.close()

def judge_model_answers_concurrently(qa_answers: List[Tuple[StandardQA, ModelAnswer]], judge_llms: List[ILLM], max_workers: int = 3) -> List[JudgeResult]:
    """使用给定的裁判大模型并发评估一批被测模型的回答
    
    Args:
        qa_answers: 问答对和对应的模型回答列表
        judge_llms: 裁判模型列表
        max_workers: 最大并发工作线程数(默认3)
        
    Returns:
        List[JudgeResult]: 评估结果列表
    """
    results = []
    
    def process_judge_task(qa: StandardQA, model_answer: ModelAnswer, judge_llm: ILLM) -> JudgeResult:
        try:
            # 检查该裁判模型是否已经评估过这个回答
            if not get_unjudged_qa_for_judge([(qa, model_answer)], judge_llm):
                return None
            
            # 使用裁判模型评估回答
            judge_result = judge_model_answer(qa, model_answer, judge_llm)
            return judge_result
        except Exception as e:
            logger.error(f"处理评估任务时出错, qa_id: {qa.id}, model_answer_id: {model_answer.id}, judge_model: {judge_llm.get_model_config().format_single_info()}, 错误: {e}", exc_info=True)
            return None
    
    # 使用线程池并发处理评估任务
    total_tasks = len(qa_answers) * len(judge_llms)
    max_workers = min(total_tasks, max_workers)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 为每个问答对和裁判模型创建评估任务
        futures = []
        for qa, model_answer in qa_answers:
            for judge_llm in judge_llms:
                futures.append(
                    executor.submit(
                        process_judge_task,
                        qa,
                        model_answer,
                        judge_llm
                    )
                )
        
        # 收集所有评估结果
        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                logger.error(f"获取评估结果时出错: {e}", exc_info=True)
    
    return results

@timer
def judge_model_answers(params: JudgeModelParams):
    """使用给定的裁判大模型评估一个被测模型的回答
    
    Args:
        params: 评估参数对象，包含模型配置、裁判模型列表等信息
    """
    # 准备裁判模型信息
    judgeModelConfigs:List[ModelConfig] = [llm.get_model_config() for llm in params.judgeLLMs]
    judgeModelIds:List[int] = [judgeModelConfig.id for judgeModelConfig in judgeModelConfigs]
    judge_model_logger_str = ", ".join([judgeModelConfig.format_single_info() for judgeModelConfig in judgeModelConfigs])
    # 准备被测模型信息
    eval_model_logger_str = params.model_config.format_single_info()
    eval_model_id = params.model_config.id # 被测模型ID
    # 用于日志
    model_logger_str = f"被测模型: {eval_model_logger_str}, 裁判模型{judge_model_logger_str}"
    # 使用ErrorCounter来记录错误次数, 
    # 如果一个裁判大模型在获取回答时出错次数超过MAX_API_ERROR_LIMIT次, 则停止评估
    errorCounter: ErrorCounter = ErrorCounter(max_errors=MAX_API_ERROR_LIMIT)

    # 获取一批问答对
    def get_next_qa_batch()->Tuple[bool, List[StandardQA]]:
        """获取下一批需要评测的问答对
        
        注意：不使用offset参数，因为SQL查询中的judged_count < %s条件会自动过滤已评估的问答对
        
        Returns:
            Tuple[bool, List[StandardQA]]: 成功标志和问答对列表
        """
        try:
            if params.qa_src == 'list':
                return True, get_unjudged_standard_qa(
                    qa_ids=params.qa_id_list, limit=params.batch_size, 
                    llm_model_id=eval_model_id, 
                    judge_llm_model_ids=judgeModelIds
                )
            else:
                return True, get_unjudged_standard_qa_batch(
                    limit=params.batch_size, 
                    llm_model_id=eval_model_id, 
                    judge_llm_model_ids=judgeModelIds
                )
        except Exception as e:
            logger.error(f"获取问答对失败! 被测模型: {eval_model_logger_str}", exc_info=True)
            return False, []

    def get_next_model_answer_batch(standardQAList:List[StandardQA])->Tuple[bool, List[Tuple[StandardQA, ModelAnswer]] ]:
        """获取下一批需要评测的模型回答
        Args:
            standardQAList: 问答对列表
        Returns:
            Tuple[bool, List[Tuple[StandardQA, ModelAnswer]]]: 成功标志和模型回答列表
        """
        try:
            qa_answers:List[Tuple[StandardQA, ModelAnswer]] = [] # 问答对和模型回答的元组列表
            # 批量获取所有问答对的模型回答
            qa_id_list = [qa.id for qa in standardQAList]
            model_answer_dict = get_saved_model_answers_batch(qa_id_list, eval_model_id)
            # 构建问答对和模型回答的元组列表
            for qa in standardQAList:
                if qa.id in model_answer_dict:
                    qa_answers.append((qa, model_answer_dict[qa.id]))
            return True, qa_answers
        except mysql.connector.Error as e:
            logger.error(f"获取模型回答失败! 被测模型: {eval_model_logger_str}, 数据库错误: {e}", exc_info=True)
            return False, []
        except Exception as e:
            logger.error(f"获取模型回答失败! 被测模型: {eval_model_logger_str}", exc_info=True)
            return False, []

    logger.info(f"开始评估模型回答: {model_logger_str}")
    # 批量获取batch_size个问答对，然后获取未被评估的候选回答，使用裁判大模型进行评估
    processed_count = 0  # 已处理的问答对数量
    while processed_count < params.limit_qa_count:
        success, standardQAList = get_next_qa_batch()
        if not success:
            logger.error(f"获取问答对失败! {model_logger_str}")
            break
        if not standardQAList:
            logger.info(f"没有已保存但未评估的回答, 停止评估. {model_logger_str}")
            break
        # 处理查询到的问答对数量，以保证不超过限制数量
        remaining_count = params.limit_qa_count - processed_count
        if len(standardQAList) > remaining_count:
            logger.info(f"查询到的问答对数量({len(standardQAList)})超过剩余限制数量({remaining_count}), 仅使用前 {remaining_count} 个")
            standardQAList = standardQAList[:remaining_count]
        # 更新已处理的问答对数量
        processed_count += len(standardQAList)
        # 问答对信息
        qa_logger_str = f"问答对 (共{len(standardQAList)}个) : {[qa.id for qa in standardQAList]}"

        # 获取需要被评估的候选回答
        logger.info(f"开始从数据库中获取需要被评估的候选回答... {model_logger_str}{qa_logger_str}")   
        success, qa_answer_tuple = get_next_model_answer_batch(standardQAList)
        if not success:
            logger.error(f"获取需要被评估的候选回答失败! 被测模型: {eval_model_logger_str}, {qa_logger_str}")
            break
        if not qa_answer_tuple:
            # 理论上，查询出来的qa都有未被评估的候选回答，所以warning
            logger.warning(f"没有找到需要评估的候选回答, 跳过当前批次. 被测模型: {eval_model_logger_str}, {qa_logger_str}")
            continue
        # 需要被评估的候选回答信息：数量，[(answerId=, modelId=, qaId=)]
        model_answer_logger_str = f"需要被评估的候选回答 (共{len(qa_answer_tuple)}个) : " \
            +f"{[(f'answerId={model_answer.id}', f'modelId={model_answer.id}', f'qaId={qa.id}') for qa, model_answer in qa_answer_tuple]}"
        
        # 使用裁判大模型对候选回答进行评估，并保存结果
        logger.info(f"开始评估候选回答并保存评估结果... {model_logger_str}, {qa_logger_str}, {model_answer_logger_str}")
        try:
            # 并发获取评估结果
            judge_results = judge_model_answers_concurrently(
                qa_answer_tuple, 
                params.judgeLLMs, 
                max_workers=params.qa_workers
            )
            
            # 批量保存评估结果
            if judge_results:
                save_evaluation_results_batch(judge_results)
                logger.info(f"评估并保存完成.{model_logger_str}, {qa_logger_str}, {model_answer_logger_str}, 评估结果数量: {len(judge_results)}")
            else:
                logger.warning(f"没有获取到有效评估结果. {model_logger_str}, {qa_logger_str}, {model_answer_logger_str}")            
            
        except Exception as e:
            logger.error(f"处理批次时出错: {e} 模型: {model_logger_str}, 问答对: {qa_logger_str}, 候选回答: {model_answer_logger_str}", exc_info=True)
            if errorCounter.count_and_check_max_error():
                logger.warning(f"错误次数超过阈值{MAX_API_ERROR_LIMIT}, 停止评估... 模型: {model_logger_str}, 问答对: {qa_logger_str}, 候选回答: {model_answer_logger_str}")
                break
            else:
                continue

def check_single_judge_result(result: dict) -> Tuple[bool, List[str], dict]:
    """检查单个裁判结果的字段完整性和结构正确性。
    
    Args:
        result: 包含裁判结果的字典，必须包含 id, model_answer_id, judge_model_id, content 字段
        
    Returns:
        Tuple[bool, List[str], dict]: 包含是否通过检查、错误信息列表和统计信息字典的元组
    """
    # 使用JudgeResultValidator进行验证
    return JudgeResultValidator.validate_judge_result_content(result['content'])

@timer
def check_judge_results_fields():
    """检查数据库中的裁判结果是否包含所有必要的字段，并且字段结构正确。"""
    # 统计信息
    stats = {
        'total_checked': 0,
        'total_errors': 0,
        'field_errors': {
            'key_points_evaluation': 0,
            'factual_errors': 0,
            'vague_statements': 0,
            'partially_correct_but_misleading': 0,
            'irrelevant_but_correct': 0
        },
        'model_errors': {},  # 记录每个模型的错误次数
        'structure_errors': {
            'key_points_evaluation': {
                'not_dict': 0,
                'missing_fields': {'missed': 0, 'partial': 0, 'matched': 0},
                'not_list': {'missed': 0, 'partial': 0, 'matched': 0}
            },
            'statement_fields': {
                'not_list': 0,
                'item_not_dict': 0,
                'missing_fields': {'exact_text': 0, 'explanation': 0},
                'not_string': {'exact_text': 0, 'explanation': 0}
            }
        }
    }
    
    BATCH_SIZE = 1000  # 每批处理的记录数
    last_checked_id = 0  # 记录最后检查的ID
    
    while True:
        conn = mysql.connector.connect(**DB_CONFIG)
        try:
            with conn.cursor(dictionary=True) as cursor:
                # 获取一批未检查的裁判结果，使用ID进行分页
                cursor.execute("""
                    SELECT id, model_answer_id, judge_model_id, content, is_eval_error
                    FROM model_eval_results
                    WHERE is_eval_error = false
                    AND id > %s
                    ORDER BY id
                    LIMIT %s
                """, (last_checked_id, BATCH_SIZE))
                results = cursor.fetchall()
                
                if not results:
                    break
                    
                logger.info(f"开始检查第 {stats['total_checked'] + 1} 到 {stats['total_checked'] + len(results)} 条裁判结果")
                
                # 收集所有需要标记为错误的裁判结果ID
                error_ids = []
                
                for result in results:
                    stats['total_checked'] += 1
                    last_checked_id = result['id']  # 更新最后检查的ID
                    
                    is_valid, errors, result_stats = check_single_judge_result(result)
                    
                    if not is_valid:
                        stats['total_errors'] += 1
                        logger.error(f"裁判结果 {result['id']} (model_answer_id={result['model_answer_id']}, judge_model_id={result['judge_model_id']}) 检查失败: {errors}")
                        error_ids.append(result['id'])
                        # 更新模型错误统计
                        model_id = result['judge_model_id']
                        stats['model_errors'][model_id] = stats['model_errors'].get(model_id, 0) + 1
                        
                        # 合并统计信息
                        for field, count in result_stats['field_errors'].items():
                            stats['field_errors'][field] += count
                        for kp_field, kp_stats in result_stats['structure_errors']['key_points_evaluation'].items():
                            if isinstance(kp_stats, dict):
                                for sub_field, count in kp_stats.items():
                                    stats['structure_errors']['key_points_evaluation'][kp_field][sub_field] += count
                            else:
                                stats['structure_errors']['key_points_evaluation'][kp_field] += kp_stats
                        for st_field, st_stats in result_stats['structure_errors']['statement_fields'].items():
                            if isinstance(st_stats, dict):
                                for sub_field, count in st_stats.items():
                                    stats['structure_errors']['statement_fields'][st_field][sub_field] += count
                            else:
                                stats['structure_errors']['statement_fields'][st_field] += st_stats
                
                # 如果有错误，批量更新数据库
                if error_ids:
                    logger.info(f"开始更新 {len(error_ids)} 条有问题的裁判结果")
                    # 构建批量更新的SQL语句
                    placeholders = ', '.join(['%s'] * len(error_ids))
                    cursor.execute(f"""
                        UPDATE model_eval_results
                        SET is_eval_error = true
                        WHERE id IN ({placeholders})
                    """, tuple(error_ids))
                    conn.commit()
                    logger.info(f"已更新 {len(error_ids)} 条裁判结果为错误状态")
                
        finally:
            conn.close()
    
    # 输出统计信息
    stats_str = "\n检查完成，统计信息如下：\n"
    stats_str += f"总共检查了 {stats['total_checked']} 条裁判结果\n"
    stats_str += f"发现 {stats['total_errors']} 条有问题的裁判结果\n\n"
    
    stats_str += "字段缺失统计：\n"
    for field, count in stats['field_errors'].items():
        stats_str += f"  {field}: {count} 次\n"
    
    stats_str += "\n结构错误统计：\n"
    stats_str += "  key_points_evaluation:\n"
    stats_str += f"    不是字典类型: {stats['structure_errors']['key_points_evaluation']['not_dict']} 次\n"
    stats_str += "    缺少必要字段:\n"
    for field, count in stats['structure_errors']['key_points_evaluation']['missing_fields'].items():
        stats_str += f"      {field}: {count} 次\n"
    stats_str += "    不是列表类型:\n"
    for field, count in stats['structure_errors']['key_points_evaluation']['not_list'].items():
        stats_str += f"      {field}: {count} 次\n"
    
    stats_str += "  statement_fields:\n"
    stats_str += f"    不是列表类型: {stats['structure_errors']['statement_fields']['not_list']} 次\n"
    stats_str += f"    项目不是字典类型: {stats['structure_errors']['statement_fields']['item_not_dict']} 次\n"
    stats_str += "    缺少必要字段:\n"
    for field, count in stats['structure_errors']['statement_fields']['missing_fields'].items():
        stats_str += f"      {field}: {count} 次\n"
    stats_str += "    不是字符串类型:\n"
    for field, count in stats['structure_errors']['statement_fields']['not_string'].items():
        stats_str += f"      {field}: {count} 次\n"
    
    stats_str += "\n模型错误统计：\n"
    for model_id, count in sorted(stats['model_errors'].items(), key=lambda x: x[1], reverse=True):
        stats_str += f"  模型ID {model_id}: {count} 次错误\n"
        
    logger.info(stats_str)
    return stats

def update_judge_result(judge_result: JudgeResult, eval_result_id:int):
    """更新数据库中的裁判评估结果。
    
    Args:
        judge_result: 裁判评估结果对象
        eval_result_id: 评估结果的数据库ID
    """
    conn = mysql.connector.connect(**DB_CONFIG)
    try:
        with conn.cursor(dictionary=True) as cursor:
            cursor.execute("""
                UPDATE model_eval_results
                SET missed_key_points = %s,
                    partial_key_points = %s,
                    matched_key_points = %s,
                    factual_errors = %s,
                    vague_statements = %s,
                    partially_correct_but_misleading = %s,
                    irrelevant_but_correct = %s,
                    reasoning_content = %s,
                    content = %s,
                    is_eval_error = false
                WHERE id = %s
            """, (
                json.dumps(judge_result.get_missed_key_points()),
                json.dumps(judge_result.get_partial_key_points()),
                json.dumps(judge_result.get_matched_key_points()),
                json.dumps(judge_result.get_factual_errors()),
                json.dumps(judge_result.get_vague_statements()),
                json.dumps(judge_result.get_partially_correct_but_misleading()),
                json.dumps(judge_result.get_irrelevant_but_correct()),
                judge_result.get_reasoning_content(),
                judge_result.get_content(),
                eval_result_id
            ))
            conn.commit()
    finally:
        conn.close()

@timer
def reprocess_error_judge_results(judgeLLMs: List[ILLM], batch_size: int = 100):
    """重新处理出错的裁判结果。
    
    Args:
        judgeLLMs: 裁判模型列表
        batch_size: 每批处理的记录数
    """
    logger.info("开始重新处理出错的裁判结果")
    
    # 使用ErrorCounter来记录错误次数
    errorCounter: ErrorCounter = ErrorCounter(max_errors=MAX_API_ERROR_LIMIT)
    
    conn = mysql.connector.connect(**DB_CONFIG)
    try:
        with conn.cursor(dictionary=True) as cursor:
            # 获取出错的裁判结果
            cursor.execute("""
                SELECT DISTINCT mer.*, mea.qa_id, mea.model_id as llm_model_id, mea.answer_content, q.question, a.answer, a.key_points
                FROM model_eval_results mer
                JOIN model_eval_answers mea ON mer.model_answer_id = mea.id
                JOIN standard_qa_relations r ON mea.qa_id = r.id
                JOIN standard_questions q ON r.question_id = q.id
                JOIN standard_answers a ON r.answer_id = a.id
                WHERE mer.is_eval_error = true
                LIMIT %s
            """, (batch_size,))
            error_results = cursor.fetchall()
            
            if not error_results:
                logger.info("没有找到出错的裁判结果")
                return
                
            logger.info(f"找到 {len(error_results)} 条出错的裁判结果")
            
            # 处理每条出错的裁判结果
            for error_result in error_results:
                try:
                    # 1. 尝试重新解析content
                    try:
                        logger.info(f"重新解析 judge_result_id={error_result['id']}, 原本数据库中出错的content: \n{error_result['content']}")
                        is_valid, errors, _ = JudgeResultValidator.validate_judge_result_content(error_result['content'])
                        
                        if is_valid:
                            old_eval_result = JudgeResult(
                                model_answer_id=error_result['model_answer_id'],
                                judge_model_id=error_result['judge_model_id'],
                                content=error_result["content"], reasoning_content=""
                            )
                            # 解析成功，更新数据库
                            logger.info(f"重新解析成功，更新数据库: model_answer_id={error_result['model_answer_id']}, judge_model_id={error_result['judge_model_id']}")
                            
                            update_judge_result(old_eval_result, error_result['id'])
                            continue
                        else:
                            logger.error(f"重新解析content失败: {errors}")
                    except Exception as e:
                        logger.error(f"重新解析content失败: {e}", exc_info=True)
                    
                    # 2. 尝试重新获取裁判结果
                    # 找到对应的裁判模型
                    judge_llm = next((llm for llm in judgeLLMs if llm.get_model_config().id == error_result['judge_model_id']), None)
                    if not judge_llm:
                        logger.error(f"找不到对应的裁判模型: judge_model_id={error_result['judge_model_id']}")
                        continue
                    
                    # 从数据库获取完整的qa信息
                    cursor.execute("""
                        SELECT r.id                 AS id,
                               q.source_question_id AS source_question_id,
                               a.source_answer_id   AS source_answer_id,
                               q.question           AS question,
                               a.answer             AS answer,
                               a.key_points         AS key_points,
                               q.status_id          AS q_status_id,
                               q.status_reason      AS q_status_reason,
                               a.status_id          AS a_status_id,
                               a.status_reason      AS a_status_reason
                        FROM standard_qa_relations r
                        JOIN standard_questions q ON r.question_id = q.id
                        JOIN standard_answers a ON r.answer_id = a.id
                        WHERE r.id = %s
                    """, (error_result['qa_id'],))
                    qa_row = cursor.fetchone()
                    if not qa_row:
                        logger.error(f"找不到对应的问答对: qa_id={error_result['qa_id']}")
                        continue
                    qa = StandardQA.from_db_row(qa_row)
                    
                    # 从数据库获取完整的model_answer信息
                    model_answer = get_saved_model_answer(error_result['qa_id'], error_result['llm_model_id'])
                    
                    if not model_answer:
                        logger.error(f"找不到对应的模型回答: model_answer_id={error_result['model_answer_id']}")
                        continue

                    # 重新获取裁判结果
                    new_judge_result = judge_model_answer(qa, model_answer, judge_llm)

                    # 检查新获取的裁判结果
                    is_valid, errors, _ = JudgeResultValidator.validate_judge_result_content(new_judge_result.get_content())

                    if is_valid and not new_judge_result.get_is_eval_error():
                        # 更新数据库
                        logger.info(f"重新获取裁判结果成功，更新数据库: model_answer_id={error_result['model_answer_id']}, judge_model_id={error_result['judge_model_id']}")
                        update_judge_result(new_judge_result, error_result['id'])
                    else:
                        logger.error(f"重新获取裁判结果仍然失败: model_answer_id={error_result['model_answer_id']}, judge_model_id={error_result['judge_model_id']}")
                        
                except Exception as e:
                    logger.error(f"处理出错的裁判结果时出错: {e}", exc_info=True)
                    if errorCounter.count_and_check_max_error():
                        logger.warning(f"错误次数超过阈值{MAX_API_ERROR_LIMIT}，停止处理")
                        break
                    continue
                    
    finally:
        conn.close()

def parse_args():
    """解析命令行参数。
    
    Returns:
        解析后的参数对象
    """
    parser = argparse.ArgumentParser(
        description='LLM问答评估工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:     
  1. 检查所有裁判结果字段完整性:
     python llm_qa_eval_judge.py --check-fields

  2. 启用Mock模式, 模型裁判大模型的回答结果:
     python llm_qa_eval_judge.py --enable-judge-mock
     
  3. 启用debug模式:
     python llm_qa_eval_judge.py --debug

  4. 重新处理出错的裁判结果:
     python llm_qa_eval_judge.py --reprocess-error
     python llm_qa_eval_judge.py --reprocess-error --enable-judge-mock
     python llm_qa_eval_judge.py --reprocess-error --debug
     python llm_qa_eval_judge.py --reprocess-error --enable-judge-mock --debug

  5. 评估：利用裁判大模型对候选回答进行评估 
  5.1 从所有问答对中评估回答:
     python llm_qa_eval_judge.py --qa-src all
  5.2 从指定问答对中评估回答:
     python llm_qa_eval_judge.py --qa-src list --qa-id-list 1
     python llm_qa_eval_judge.py --qa-src list --qa-id-list 1,2,3
  5.3 调整并发和批处理参数: 同时并发评估3个被测模型, 限制每个模型只评估20个问题, 每批从数据库中获取的问题数量为10, 每个模型同时被并发评估的问题数量为2
     python llm_qa_eval_judge.py --model-workers 3 --limit-qa-count 20 --batch-size 10 --qa-workers 2 
  5.4 组合使用:
     python llm_qa_eval_judge.py --model-workers 3 --limit-qa-count 202 --batch-size 100 --qa-workers 5 --qa-src all
     python llm_qa_eval_judge.py --model-workers 1 --limit-qa-count 1 --batch-size 1 --qa-workers 1 --qa-src list --qa-id-list 5
     python llm_qa_eval_judge.py --model-workers 3 --limit-qa-count 202 --batch-size 100 --qa-workers 5 --qa-src all --enable-judge-mock --debug
"""
    )
    
    # 默认参数
    DEFAULT_MODEL_WORKERS = 3
    DEFAULT_LIMIT_QA_COUNT = 8
    DEFAULT_BATCH_SIZE = 8
    DEFAULT_QA_WORKERS = 2
    DEFAULT_QA_SRC = 'all'

    # 模型并发数
    parser.add_argument(
        '--model-workers',
        type=int,
        default=DEFAULT_MODEL_WORKERS,
        help=f'同时并发评估的被测模型数量 (默认: {DEFAULT_MODEL_WORKERS})'
    )
    
    # 问答对并发数
    parser.add_argument(
        '--qa-workers',
        type=int,
        default=DEFAULT_QA_WORKERS,
        help=f'裁判模型并发评估任务的数量 (默认: {DEFAULT_QA_WORKERS})'
    )
    
    # 批处理大小
    parser.add_argument(
        '--batch-size',
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f'每批从数据库中获取的问答对数量 (默认: {DEFAULT_BATCH_SIZE})'
    )
    
    # 限制评估的问答对数量
    parser.add_argument(
        '--limit-qa-count',
        type=int,
        default=DEFAULT_LIMIT_QA_COUNT,
        help=f'限制评估的问答对数量 (默认: {DEFAULT_LIMIT_QA_COUNT})'
    )

    # 问答对来源
    parser.add_argument(
        '--qa-src',
        type=str,
        choices=['all', 'list'],
        default=DEFAULT_QA_SRC,
        help=f'问答对来源: all 表示选择所有 (active or modified-active or pending-active) ；list 表示从qa_id_list作为问答来源 (默认: {DEFAULT_QA_SRC})'
    )

    # 问答对ID列表
    parser.add_argument(
        '--qa-id-list',
        type=str,
        help='问答对ID列表, 用逗号分隔, 仅在qa-src=list时有效'
    )

    # 重新处理出错的裁判结果
    parser.add_argument(
        '--reprocess-error',
        action='store_true',
        help='重新处理出错的裁判结果'
    )
    
    # 检查裁判结果字段完整性
    parser.add_argument(
        '--check-fields',
        action='store_true',
        help='检查裁判结果字段完整性'
    )
    
    # 启用LLM Mock模式
    parser.add_argument(
        '--enable-judge-mock',
        action='store_true',
        help='启用LLM Mock模式(仅用于测试), 模型裁判大模型的回答结果会被mock为固定值; 默认关闭'
    )
    
    # 调试模式
    parser.add_argument(
        '--debug',
        action='store_true',
        help='启用调试模式，包括使用秒级时间戳记录日志等; 默认关闭'
    )
    
    return parser.parse_args()

@timer
def main():
    try:
        # 解析命令行参数
        args = parse_args()

        # 配置日志器
        global logger
        logger = setup_logger(
            log_file=LOG_FILE,
            name=__name__,
            add_timestamp=True,
            timestamp_type=TimestampType.SECOND if args.debug else TimestampType.DAY,
            print_log_location=True
        )
        
        # 如果是检查字段完整性
        if args.check_fields:
            check_judge_results_fields()
            return
            
        # 处理问答对来源参数
        qa_src = args.qa_src
        qa_id_list = []
        if qa_src == 'list':
            if not args.qa_id_list:
                raise ValueError("使用 --qa-src list 时必须指定 --qa-id-list")
            qa_id_list = [int(id) for id in args.qa_id_list.split(',')]
        
        # 输出运行参数
        logger.info("运行参数配置: ")
        if args.debug:
            logger.warning("运行模式: Debug")
        logger.info(f"  同时并发同时并发评估的被测模型数量(model-workers): {args.model_workers}")
        logger.info(f"  限制每个被测模型评估的问题总数(limit-qa-count): {args.limit_qa_count}")
        logger.info(f"  每批从数据库中获取的问答对数量(batch-size): {args.batch_size}")
        logger.info(f"  每个被测模型同时并发评估的问题数量(qa-workers): {args.qa_workers}")
        logger.info(f"  问答对来源(qa-src): {qa_src}")
        if qa_src == 'list':
            logger.info(f"  问答对ID列表: {qa_id_list}")
        
        # 被测模型配置文件路径
        EVAL_MODELS_CONFIG_FILE_PATH = CONFIG[CONFIG_NAME]['eval_models_config_file_path']
        ABS_EVAL_MODELS_CONFIG_FILE_PATH = os.path.abspath(EVAL_MODELS_CONFIG_FILE_PATH)
        # 裁判模型配置文件路径
        JUDGE_MODELS_CONFIG_FILE_PATH = CONFIG[CONFIG_NAME]['judge_models_config_file_path']
        ABS_JUDGE_MODELS_CONFIG_FILE_PATH = os.path.abspath(JUDGE_MODELS_CONFIG_FILE_PATH)

        # 初始化API密钥加密器
        logger.info(f"初始化API加解密器")
        apiKeyEncryptor = ApiKeyEncryptor(get_api_key_secret_key(CONFIG))
        
        # 加载初始化所有裁判模型
        logger.info(f"正在从 {ABS_JUDGE_MODELS_CONFIG_FILE_PATH} 加载裁判模型配置")
        jModelConfigs = load_models_configs(
            apiKeyEncryptor=apiKeyEncryptor,
            db_config=DB_CONFIG,
            file_path=ABS_JUDGE_MODELS_CONFIG_FILE_PATH
        )
        logger.info(f"已加载 {len(jModelConfigs)} 个裁判模型配置: {[jModelConfig.model_name for jModelConfig in jModelConfigs]}")
        judgeLLMs:List[ILLM] = []
        if args.enable_judge_mock:
            logger.warning(f"已启用Mock模式, 裁判大模型的推理会被mock为固定值:\n{MOCK_JUDGE_REASONING}")
            logger.warning(f"已启用Mock模式, 裁判大模型的回答会被mock为固定值:\n{MOCK_JUDGE_RESULT}")
            mockConfigs:List[ModelConfig] = []
            for jModelConfig in jModelConfigs:
                mockConfig:ModelConfig = ModelConfig(
                    id=jModelConfig.id,
                    model_name=f"{jModelConfig.model_name}-mock",
                    model_id=jModelConfig.model_id,
                    api_key=jModelConfig.api_key,
                    base_url=jModelConfig.base_url,
                    model_remark=f"[Mock]{jModelConfig.model_remark}",
                    api_type="Mock",
                    model_type=jModelConfig.modelType.value,
                    new_model=jModelConfig.is_new_model(),
                )
                mockConfigs.append(mockConfig)
                mockLLM:ILLM = MockLLM(
                    modelConfig=mockConfig,
                    mock_content=MOCK_JUDGE_RESULT,
                    mock_reasoning=MOCK_JUDGE_REASONING,
                    )
                judgeLLMs.append(mockLLM)
            logger.info(f"已初始化 {len(mockConfigs)} 个Mock裁判模型: {[jModelConfig.model_name for jModelConfig in mockConfigs]}")
        else:
            judgeLLMs:ILLM = [LLM(config) for config in jModelConfigs]
            logger.info(f"已初始化 {len(judgeLLMs)} 个裁判模型: {[jModelConfig.model_name for jModelConfig in jModelConfigs]}")
        
        # 如果是重新处理出错的裁判结果
        if args.reprocess_error:
            reprocess_error_judge_results(judgeLLMs, args.batch_size)
            return
        
        # 加载被测模型配置
        logger.info(f"正在从 {ABS_EVAL_MODELS_CONFIG_FILE_PATH} 加载被测模型配置")
        evalModelConfigList = load_models_configs(
            apiKeyEncryptor=apiKeyEncryptor,
            db_config=DB_CONFIG,
            file_path=ABS_EVAL_MODELS_CONFIG_FILE_PATH
        )
        logger.info(f"已加载 {len(evalModelConfigList)} 个被测模型配置: {[sModelConfig.model_name for sModelConfig in evalModelConfigList]}")
        
        logger.info(f"裁判模型的系统提示词:\n{JUDGE_SYSTEM_PROMPT}")
        logger.info("开始进行并发评估......")
        
        # 使用线程池并发评估多个模型
        # 优化max_workers设置：取用户指定的worker数量和实际模型数量的最小值
        max_workers = min(args.model_workers, len(evalModelConfigList))
        logger.info(f"线程池配置: 用户指定worker数={args.model_workers}, 实际模型数={len(evalModelConfigList)}, 最终max_workers={max_workers}")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for sModelConfig in evalModelConfigList:
                futures.append(
                    executor.submit(
                        judge_model_answers,
                        JudgeModelParams(
                            model_config=sModelConfig,
                            judgeLLMs=judgeLLMs,
                            qa_workers=args.qa_workers,
                            batch_size=args.batch_size,
                            limit_qa_count=args.limit_qa_count,
                            qa_src=qa_src,
                            qa_id_list=qa_id_list,
                            api_key_encryptor=apiKeyEncryptor
                        )
                    )
                )
            
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"模型评估过程中出错: {e}", exc_info=True)
                    
    except Exception as e:
        logger.error(f"主程序执行出错: {e}", exc_info=True)

if __name__ == "__main__":
    main()