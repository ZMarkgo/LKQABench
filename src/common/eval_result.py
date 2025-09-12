import mysql.connector
import json
from typing import List, Dict
from dataclasses import dataclass, field
from common.config import load_config, get_db_config
from common.entity import QAStatusEnum
import math
from rouge_score import rouge_scorer
import nltk
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from datetime import datetime, date

config = load_config()

# 获取数据库配置
db_config = get_db_config(config)

@dataclass
class TopicInfo:
    topic_id: int
    topic_name: str
    topic_description: str

    @classmethod
    def from_db(cls, db_row: dict):
        return cls(
            topic_id=db_row['topic_id'],
            topic_name=db_row['topic_name'],
            topic_description=db_row['topic_description']
        )

@dataclass
class JudgeEvalResult:
    missed_key_points: List[int]
    partial_key_points: List[int]
    matched_key_points: List[int]

@dataclass
class EvalResult:
    # StandardQA字段
    qa_id: int
    split_question: str
    split_answer: str
    key_points: List[str]
    qa_status: QAStatusEnum

    # 主题分类和认知水平字段
    topics: List[TopicInfo]  # 修改为列表，因为一个问题可以有多个主题
    cognitive_level: int
    cognitive_level_name: str
    cognitive_level_description: str

    # ModelAnswer字段
    model_answer_id: int
    llm_model_id: int
    model_name: str
    model_id: str
    model_remark: str
    reasoning_content: str
    answer_content: str

    # ModelEvalResult字段
    eval_id: int

    missed_key_points: List[int]
    partial_key_points: List[int]
    matched_key_points: List[int]

    factual_errors_merged: List[Dict[str, str]]
    vague_statements_merged: List[Dict[str, str]]
    partially_correct_merged: List[Dict[str, str]]
    irrelevant_correct_merged: List[Dict[str, str]]
    factual_errors_unmerged: List[Dict[str, str]]
    vague_statements_unmerged: List[Dict[str, str]]
    partially_correct_unmerged: List[Dict[str, str]]
    irrelevant_correct_unmerged: List[Dict[str, str]]
    factual_errors_unmatched: List[Dict[str, str]]
    vague_statements_unmatched: List[Dict[str, str]]
    partially_correct_unmatched: List[Dict[str, str]]
    irrelevant_correct_unmatched: List[Dict[str, str]]
    extra_key_points: List[Dict[str, str]]
    missed_key_points_details: List[Dict[str, str]]

    factual_errors: List[str] = field(default_factory=list)
    vague_statements: List[str] = field(default_factory=list)
    partially_correct_but_misleading: List[str] = field(default_factory=list)
    irrelevant_but_correct: List[str] = field(default_factory=list)
    factual_errors_ume: List[str] = field(default_factory=list)
    vague_statements_ume: List[str] = field(default_factory=list)
    partially_correct_but_misleading_ume: List[str] = field(default_factory=list)
    irrelevant_but_correct_ume: List[str] = field(default_factory=list)
    
    factual_errors_ume_llm_counts: List[int] = field(default_factory=list)
    vague_statements_ume_llm_counts: List[int] = field(default_factory=list)
    partially_correct_but_misleading_ume_llm_counts: List[int] = field(default_factory=list)
    irrelevant_but_correct_ume_llm_counts: List[int] = field(default_factory=list)

    # 得分
    coverage_score: float = 0.0  # 关键知识点覆盖度得分
    accuracy_score: float = 0.0  # 正确性得分
    clarity_score: float = 0.0   # 清晰度得分
    total_score: float = 0.0     # 总分

    raw_accuracy_score: float = 0.0,
    raw_clarity_score: float = 0.0,

    # 新增字段：多个打分模型的评估结果
    judge_evals: Dict[int, JudgeEvalResult] = field(default_factory=dict)

    is_version_specific: int = 0
    source_question_id: int = None
    source_ml_id: int = None
    qa_date: datetime = None
    model_release_date: datetime = None

    def get_score_breakdown(self):
        """返回评分的详细分解"""
        return {
            "coverage_score": self.coverage_score,
            "accuracy_score": self.accuracy_score,
            "clarity_score": self.clarity_score,
            "total_score": self.total_score,
            "matched_key_points_count": len(self.matched_key_points),
            "partial_key_points_count": len(self.partial_key_points),
            "missed_key_points_count": len(self.missed_key_points),
            "factual_errors_count": len(self.factual_errors),
            "misleading_count": len(self.partially_correct_but_misleading),
            "vague_statements_count": len(self.vague_statements),
            "irrelevant_count": len(self.irrelevant_but_correct)
        }

    @classmethod
    def from_db(cls, db_row: dict, topics: List[TopicInfo], judge_evals: Dict[int, JudgeEvalResult]):
        """从数据库记录构建EvalResult实例"""
        missed_key_points = json.loads(db_row['missed_key_points'])
        partial_key_points = json.loads(db_row['partial_key_points'])
        matched_key_points = json.loads(db_row['matched_key_points'])
        factual_errors_merged = json.loads(db_row['factual_errors_merged'])
        vague_statements_merged = json.loads(db_row['vague_statements_merged'])
        partially_correct_merged = json.loads(db_row['partially_correct_merged'])
        irrelevant_correct_merged = json.loads(db_row['irrelevant_correct_merged'])
        factual_errors_unmerged = json.loads(db_row['factual_errors_unmerged'])
        vague_statements_unmerged = json.loads(db_row['vague_statements_unmerged'])
        partially_correct_unmerged = json.loads(db_row['partially_correct_unmerged'])
        irrelevant_correct_unmerged = json.loads(db_row['irrelevant_correct_unmerged'])
        factual_errors_unmatched = json.loads(db_row['factual_errors_unmatched'])
        vague_statements_unmatched = json.loads(db_row['vague_statements_unmatched'])
        partially_correct_unmatched = json.loads(db_row['partially_correct_unmatched'])
        irrelevant_correct_unmatched = json.loads(db_row['irrelevant_correct_unmatched'])
        extra_key_points = json.loads(db_row['extra_key_points'])
        missed_key_points_details = json.loads(db_row['missed_key_points_details'])

        factual_errors = [item['exact_text'] for item in factual_errors_merged]
        vague_statements = [item['exact_text'] for item in vague_statements_merged]
        partially_correct_but_misleading = [item['exact_text'] for item in partially_correct_merged]
        irrelevant_but_correct = [item['exact_text'] for item in irrelevant_correct_merged]
        factual_errors_ume = [item['exact_text'] for item in factual_errors_unmerged]
        vague_statements_ume = [item['exact_text'] for item in vague_statements_unmerged]
        partially_correct_but_misleading_ume = [item['exact_text'] for item in partially_correct_unmerged]
        irrelevant_but_correct_ume = [item['exact_text'] for item in irrelevant_correct_unmerged]

        factual_errors_ume_llm_counts = [item['count'] for item in factual_errors_unmerged]
        vague_statements_ume_llm_counts = [item['count'] for item in vague_statements_unmerged]
        partially_correct_but_misleading_ume_llm_counts = [item['count'] for item in partially_correct_unmerged]
        irrelevant_but_correct_ume_llm_counts = [item['count'] for item in irrelevant_correct_unmerged]

        instance = cls(
            # StandardQA字段
            qa_id=db_row['qa_id'],
            split_question=db_row['split_question'],
            split_answer=db_row['split_answer'],
            key_points=json.loads(db_row['key_points']),
            qa_status=QAStatusEnum(db_row['status_id']),
            source_question_id=db_row['source_question_id'],
            source_ml_id=db_row['source_ml_id'],

            topics=topics,
            cognitive_level=db_row['cognitive_level'],
            cognitive_level_name=db_row['cognitive_level_name'],
            cognitive_level_description=db_row['cognitive_level_description'],

            # ModelAnswer字段
            model_answer_id=db_row['model_answer_id'],
            llm_model_id=db_row['llm_model_id'],
            model_name=db_row['model_name'],
            model_id=db_row['model_id'],
            model_remark=db_row['model_remark'],
            reasoning_content=db_row['reasoning_content'],
            answer_content=db_row['answer_content'],

            # ModelEvalResult字段
            eval_id=db_row['eval_id'],
            missed_key_points=missed_key_points,
            partial_key_points=partial_key_points,
            matched_key_points=matched_key_points,
            factual_errors_merged=factual_errors_merged,
            vague_statements_merged=vague_statements_merged,
            partially_correct_merged=partially_correct_merged,
            irrelevant_correct_merged=irrelevant_correct_merged,
            factual_errors_unmerged=factual_errors_unmerged,
            vague_statements_unmerged=vague_statements_unmerged,
            partially_correct_unmerged=partially_correct_unmerged,
            irrelevant_correct_unmerged=irrelevant_correct_unmerged,
            factual_errors_unmatched=factual_errors_unmatched,
            vague_statements_unmatched=vague_statements_unmatched,
            partially_correct_unmatched=partially_correct_unmatched,
            irrelevant_correct_unmatched=irrelevant_correct_unmatched,
            extra_key_points=extra_key_points,
            missed_key_points_details=missed_key_points_details,

            factual_errors=factual_errors,
            vague_statements=vague_statements,
            partially_correct_but_misleading=partially_correct_but_misleading,
            irrelevant_but_correct=irrelevant_but_correct,
            factual_errors_ume=factual_errors_ume,
            vague_statements_ume=vague_statements_ume,
            partially_correct_but_misleading_ume=partially_correct_but_misleading_ume,
            irrelevant_but_correct_ume=irrelevant_but_correct_ume,

            factual_errors_ume_llm_counts=factual_errors_ume_llm_counts,
            vague_statements_ume_llm_counts=vague_statements_ume_llm_counts,
            partially_correct_but_misleading_ume_llm_counts=partially_correct_but_misleading_ume_llm_counts,
            irrelevant_but_correct_ume_llm_counts=irrelevant_but_correct_ume_llm_counts,

            judge_evals=judge_evals,
            is_version_specific = db_row['is_version_specific'],
            qa_date = db_row['qa_date'],
            model_release_date = datetime.combine(db_row['model_release_date'], datetime.min.time())
        )
        
        # 计算评分
        # instance.calculate_scores()
        instance.calculate_scores(coverage_score_method=2, unmerged_errors_valid=False, accuracy_and_clarity_positive=True)
        
        return instance
    
    def calculate_coverage_score_before_fusion(self, coverage_weight):
        total_key_points = len(self.key_points)
        coverage_scores = []

        if total_key_points > 0 and self.judge_evals:
            base_point = coverage_weight / total_key_points
            for je in self.judge_evals.values():
                matched_score = len(je.matched_key_points) * base_point
                partial_score = len(je.partial_key_points) * (base_point / 2)
                coverage_scores.append(matched_score + partial_score)
            self.coverage_score = sum(coverage_scores) / len(coverage_scores)
        else:
            # 没有关键点或没有judge数据，按默认方式处理
            self.calculate_coverage_score_after_fusion(coverage_weight)

    def calculate_coverage_score_after_fusion(self, coverage_weight):
        total_key_points = len(self.key_points)
        matched_key_points = len(self.matched_key_points)
        if total_key_points > 0:
            if matched_key_points == total_key_points:
                self.coverage_score = coverage_weight
            else:
                base_point = coverage_weight / total_key_points
                matched_score = len(self.matched_key_points) * base_point
                partial_score = len(self.partial_key_points) * (base_point / 2)
                self.coverage_score = matched_score + partial_score # 直接修改这行没用，即使每个模型对每个回答的分数拉开了，平均下来也没有效果
                
        else:
            self.coverage_score = coverage_weight  # 如果没有关键点，则默认满分
            
    def calculate_pass_coverage_score(self, coverage_weight):
        total_key_points = len(self.key_points)
        matched_key_points = 0
        partial_key_points = 0

        if total_key_points > 0:
            total_key_points = 0
            for je in self.judge_evals.values():
                matched_key_points += len(je.matched_key_points)
                partial_key_points += len(je.partial_key_points)
                total_key_points += len(self.key_points)
            # total_key_points = len(self.key_points)
            # matched_key_points = len(self.matched_key_points)
            # partial_key_points = len(self.partial_key_points)
            if matched_key_points == total_key_points:
                self.coverage_score = coverage_weight
            else:
                k = 2
                expand_rate = math.ceil(4/total_key_points)
                total_key_points_expand = total_key_points * expand_rate
                covarage_key_points_expand = (matched_key_points + partial_key_points / 2) * expand_rate
                self.coverage_score = coverage_weight * self.real_comb(covarage_key_points_expand, k) / self.real_comb(total_key_points_expand, k)
        else:
            self.coverage_score = coverage_weight  # 如果没有关键点，则默认满分

    def real_comb(self, x, k):
        if x < k:
            return 0  # 一般定义下 x < k 时组合数为0
        return math.gamma(x + 1) / (math.gamma(k + 1) * math.gamma(x - k + 1))

    def calculate_rouge_l_coverage_score(self, coverage_weight):
        reference = self.split_answer
        hypothesis = self.answer_content

        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        score = scorer.score(reference, hypothesis)

        # 返回 F1 分数（也可以返回 precision/recall）
        self.coverage_score = score['rougeL'].fmeasure * coverage_weight
        
    def calculate_meteor_coverage_score(self, coverage_weight):
        reference = self.split_answer
        hypothesis = self.answer_content
            
        reference_tokens = word_tokenize(reference)
        hypothesis_tokens = word_tokenize(hypothesis)

        self.coverage_score = meteor_score([reference_tokens], hypothesis_tokens) * coverage_weight

    def calculate_scores(self, 
                         coverage_weight: float = 50.0, 
                         accuracy_weight: float = 30.0, 
                         clarity_weight: float = 20.0,
                         factual_error_penalty: float = 15.0,
                         misleading_penalty: float = 10.0,
                         vague_penalty: float = 8.0,
                         irrelevant_penalty: float = 5.0,
                         coverage_score_method: int = 2,
                         unmerged_errors_valid: bool = False,
                         accuracy_and_clarity_positive: bool = True):
        """
        计算模型回答的评分，结果存储在成员变量中
        
        参数:
            coverage_weight: 关键知识点覆盖度权重，默认50分
            accuracy_weight: 正确性权重，默认30分
            clarity_weight: 清晰度权重，默认20分
            factual_error_penalty: 每个事实错误的惩罚分数，较大惩罚
            misleading_penalty: 每个误导性内容的惩罚分数，较大惩罚
            vague_penalty: 每个模糊表述的惩罚分数，中等惩罚
            irrelevant_penalty: 每个无关内容的惩罚分数，较小惩罚
            coverage_score_method: 覆盖率结果计算方法
            unmerged_errors_valid: 对未融合的结果列表进行惩罚
            accuracy_and_clarity_positive: 保证正确性与清晰度为正
        """
        if (coverage_weight + accuracy_weight + clarity_weight) != 100.0:
            raise ValueError("权重总和必须为100")
        # 计算关键知识点覆盖度得分
        
        if (coverage_score_method == 0):
            self.calculate_coverage_score_before_fusion(coverage_weight)
        elif (coverage_score_method == 1):
            self.calculate_pass_coverage_score(coverage_weight)
        elif (coverage_score_method == 2):
            self.calculate_coverage_score_after_fusion(coverage_weight)
        elif (coverage_score_method == 3):
            self.calculate_rouge_l_coverage_score(coverage_weight)
        else:
            self.calculate_meteor_coverage_score(coverage_weight)
        
        # 计算正确性得分
        accuracy_penalty = (len(self.factual_errors) * factual_error_penalty + 
                           len(self.partially_correct_but_misleading) * misleading_penalty)
        
        if (unmerged_errors_valid):
            accuracy_penalty += (sum(self.factual_errors_ume_llm_counts) * factual_error_penalty + 
                           sum(self.partially_correct_but_misleading_ume_llm_counts) * misleading_penalty) / len(self.judge_evals) # 扣单个模型的分，如果以后要考虑到count的话这里需要改
        
        self.raw_accuracy_score = accuracy_weight - accuracy_penalty
        if (accuracy_and_clarity_positive):
            self.accuracy_score = max(0, self.raw_accuracy_score)
        else:
            self.accuracy_score = self.raw_accuracy_score
        
        # 计算清晰度得分
        clarity_penalty = (len(self.vague_statements) * vague_penalty + 
                          len(self.irrelevant_but_correct) * irrelevant_penalty)
        if (unmerged_errors_valid):
            clarity_penalty += (sum(self.vague_statements_ume_llm_counts) * vague_penalty + 
                          sum(self.irrelevant_but_correct_ume_llm_counts) * irrelevant_penalty) / len(self.judge_evals)
        
        self.raw_clarity_score = clarity_weight - clarity_penalty
        if (accuracy_and_clarity_positive):
            self.clarity_score = max(0, self.raw_clarity_score)
        else:
            self.clarity_score = self.raw_clarity_score
        
        # 计算总分
        self.total_score = self.coverage_score + self.accuracy_score + self.clarity_score

def get_EvalResult_by_result_id(result_id) -> EvalResult:
    """从数据库中提取指定的evalresult"""
    # 创建数据库连接
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor(dictionary=True)

    try:
        # 通过result_id查询评测结果，并连接相关表获取完整信息
        query = """
            SELECT r.id                          AS qa_id,
                q.question                    AS split_question,
                q.source_question_id          AS source_question_id,
                q.source_ml_id                AS source_ml_id,
                a.answer                      AS split_answer,
                a.key_points                  AS key_points,
                a.status_id                   AS status_id,
                mea.id                        AS model_answer_id,
                mea.model_id                  AS llm_model_id,
                m.model_name                  AS model_name,
                m.model_id                    AS model_id,
                m.model_remark                AS model_remark,
                m.release_date                AS model_release_date,
                mea.reasoning_content         AS reasoning_content,
                mea.answer_content            AS answer_content,
                aer.id                        AS eval_id,
                aer.missed_key_points         AS missed_key_points,
                aer.partial_key_points        AS partial_key_points,
                aer.matched_key_points        AS matched_key_points,
                aer.factual_errors_merged     AS factual_errors_merged,
                aer.vague_statements_merged   AS vague_statements_merged,
                aer.partially_correct_merged  AS partially_correct_merged,
                aer.irrelevant_correct_merged AS irrelevant_correct_merged,
                aer.factual_errors_unmerged   AS factual_errors_unmerged,
                aer.vague_statements_unmerged AS vague_statements_unmerged,
                aer.partially_correct_unmerged      AS partially_correct_unmerged,
                aer.irrelevant_correct_unmerged     AS irrelevant_correct_unmerged,
                aer.factual_errors_unmatched        AS factual_errors_unmatched,
                aer.vague_statements_unmatched      AS vague_statements_unmatched,
                aer.partially_correct_unmatched     AS partially_correct_unmatched,
                aer.irrelevant_correct_unmatched    AS irrelevant_correct_unmatched,
                aer.extra_key_points          AS extra_key_points,
                aer.missed_key_points_details AS missed_key_points_details,
                cl.level                      AS cognitive_level,
                cl.name                       AS cognitive_level_name,
                cl.description                AS cognitive_level_description,
                vsa.is_version_specific       AS is_version_specific,
                COALESCE(ml.first_email_date, oa.creation_date)         AS qa_date
            FROM aggregated_eval_results AS aer
                    JOIN model_eval_answers mea ON aer.model_answer_id = mea.id
                    JOIN llm_models m ON m.id = mea.model_id
                    JOIN standard_qa_relations r ON r.id = mea.qa_id
                    JOIN standard_questions q ON r.question_id = q.id
                    JOIN standard_answers a ON r.answer_id = a.id
                    JOIN standard_question_cognitive_levels sqcl ON q.id = sqcl.standard_question_id
                    JOIN cognitive_levels cl ON cl.id = sqcl.cognitive_level_id
                    JOIN version_specific_analysis vsa ON vsa.qa_relation_id = r.id
                    JOIN dataset_maps dm ON dm.qa_id = r.id
                    LEFT JOIN mailing_lists ml ON ml.id = a.source_ml_id
                    LEFT JOIN original_answers oa ON oa.id = a.source_answer_id
            WHERE aer.id = %s
        """
        cursor.execute(query, (result_id,))
        row = cursor.fetchone()

        if not row:
            return None
        
        # 然后获取该问题的所有主题分类
        topics_query = """
            SELECT tc.id as topic_id, 
                   tc.name as topic_name, 
                   tc.description as topic_description
            FROM standard_questions q
                JOIN standard_qa_relations r ON r.question_id = q.id
                JOIN model_eval_answers mea ON mea.qa_id = r.id
                JOIN aggregated_eval_results aer ON aer.model_answer_id = mea.id
                JOIN standard_question_topics sqt ON q.id = sqt.standard_question_id
                JOIN topic_categories tc ON sqt.topic_id = tc.id
            WHERE aer.id = %s
        """
        cursor.execute(topics_query, (result_id,))
        topics = [
            TopicInfo.from_db(topic_row)
            for topic_row in cursor.fetchall()
        ]

        return EvalResult.from_db(row, topics)

    finally:
        cursor.close()
        conn.close()

def get_all_EvalResult() -> List[EvalResult]:
    """从数据库获取所有测评结果"""
    # 创建数据库连接
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor(dictionary=True)

    try:
        # 先查 model_eval_results 表，构建 judge_evals 映射
        judge_query = """
            SELECT model_answer_id,
                   judge_model_id,
                   missed_key_points,
                   partial_key_points,
                   matched_key_points
            FROM model_eval_results
        """
        cursor.execute(judge_query)
        judge_rows = cursor.fetchall()

        from collections import defaultdict
        import json

        judge_evals_map: Dict[int, Dict[int, JudgeEvalResult]] = defaultdict(dict)

        for row in judge_rows:
            model_answer_id = row['model_answer_id']
            judge_model_id = row['judge_model_id']
            judge_evals_map[model_answer_id][judge_model_id] = JudgeEvalResult(
                missed_key_points=json.loads(row['missed_key_points']),
                partial_key_points=json.loads(row['partial_key_points']),
                matched_key_points=json.loads(row['matched_key_points'])
            )

        # 查询所有评测结果
        query = """
            SELECT r.id                          AS qa_id,
                q.question                    AS split_question,
                q.source_question_id          AS source_question_id,
                q.source_ml_id                AS source_ml_id,
                a.answer                      AS split_answer,
                a.key_points                  AS key_points,
                a.status_id                   AS status_id,
                mea.id                        AS model_answer_id,
                mea.model_id                  AS llm_model_id,
                m.model_name                  AS model_name,
                m.model_id                    AS model_id,
                m.model_remark                AS model_remark,
                m.release_date                AS model_release_date,
                mea.reasoning_content         AS reasoning_content,
                mea.answer_content            AS answer_content,
                aer.id                        AS eval_id,
                aer.missed_key_points         AS missed_key_points,
                aer.partial_key_points        AS partial_key_points,
                aer.matched_key_points        AS matched_key_points,
                aer.factual_errors_merged     AS factual_errors_merged,
                aer.vague_statements_merged   AS vague_statements_merged,
                aer.partially_correct_merged  AS partially_correct_merged,
                aer.irrelevant_correct_merged AS irrelevant_correct_merged,
                aer.factual_errors_unmerged   AS factual_errors_unmerged,
                aer.vague_statements_unmerged AS vague_statements_unmerged,
                aer.partially_correct_unmerged      AS partially_correct_unmerged,
                aer.irrelevant_correct_unmerged     AS irrelevant_correct_unmerged,
                aer.factual_errors_unmatched   AS factual_errors_unmatched,
                aer.vague_statements_unmatched AS vague_statements_unmatched,
                aer.partially_correct_unmatched      AS partially_correct_unmatched,
                aer.irrelevant_correct_unmatched     AS irrelevant_correct_unmatched,
                aer.extra_key_points          AS extra_key_points,
                aer.missed_key_points_details AS missed_key_points_details,
                cl.level                      AS cognitive_level,
                cl.name                       AS cognitive_level_name,
                cl.description                AS cognitive_level_description,
                vsa.is_version_specific       AS is_version_specific,
                COALESCE(ml.first_email_date, oa.creation_date)         AS qa_date
            FROM aggregated_eval_results AS aer
                    JOIN model_eval_answers mea ON aer.model_answer_id = mea.id
                    JOIN llm_models m ON m.id = mea.model_id
                    JOIN standard_qa_relations r ON r.id = mea.qa_id
                    JOIN standard_questions q ON r.question_id = q.id
                    JOIN standard_answers a ON r.answer_id = a.id
                    JOIN standard_question_cognitive_levels sqcl ON q.id = sqcl.standard_question_id
                    JOIN cognitive_levels cl ON cl.id = sqcl.cognitive_level_id
                    JOIN version_specific_analysis vsa ON vsa.qa_relation_id = r.id
                    LEFT JOIN mailing_lists ml ON ml.id = a.source_ml_id
                    LEFT JOIN original_answers oa ON oa.id = a.source_answer_id
        """
        cursor.execute(query)
        rows = cursor.fetchall()

        evalResultList = []

        for row in rows:
            # 对每个评测结果获取其所有主题分类
            topics_query = """
                SELECT tc.id as topic_id, 
                       tc.name as topic_name, 
                       tc.description as topic_description
                FROM standard_questions q
                    JOIN standard_qa_relations r ON r.question_id = q.id
                    JOIN model_eval_answers mea ON mea.qa_id = r.id
                    JOIN aggregated_eval_results aer ON aer.model_answer_id = mea.id
                    JOIN standard_question_topics sqt ON q.id = sqt.standard_question_id
                    JOIN topic_categories tc ON sqt.topic_id = tc.id
                WHERE aer.id = %s
            """
            cursor.execute(topics_query, (row['eval_id'],))
            topics = [
                TopicInfo.from_db(topic_row)
                for topic_row in cursor.fetchall()
            ]

            model_answer_id = row['model_answer_id']
            judge_evals = judge_evals_map.get(model_answer_id, {})
            eval_result = EvalResult.from_db(row, topics, judge_evals)
            evalResultList.append(eval_result)


                

        return evalResultList

    finally:
        cursor.close()
        conn.close()

def get_EvalResult(model_ids: List[int] = [], dataset_version: int = 3, dataset_source: int = 0) -> List[EvalResult]:
    """
    从数据库获取所有测评结果
    
    参数：
        model_ids：指定被测模型的结果（默认为全部）  

        dataset_version：指定数据集版本（默认为3，最新）  

        dataset_source：指定问答对来源，0:全部，1:来自QA，2:来自邮件
    """
    # 创建数据库连接
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor(dictionary=True)

    try:

        # 先查 model_eval_results 表，构建 judge_evals 映射
        judge_query = """
            SELECT model_answer_id,
                   judge_model_id,
                   missed_key_points,
                   partial_key_points,
                   matched_key_points
            FROM model_eval_results
        """
        cursor.execute(judge_query)
        judge_rows = cursor.fetchall()

        from collections import defaultdict
        import json

        judge_evals_map: Dict[int, Dict[int, JudgeEvalResult]] = defaultdict(dict)

        for row in judge_rows:
            model_answer_id = row['model_answer_id']
            judge_model_id = row['judge_model_id']
            judge_evals_map[model_answer_id][judge_model_id] = JudgeEvalResult(
                missed_key_points=json.loads(row['missed_key_points']),
                partial_key_points=json.loads(row['partial_key_points']),
                matched_key_points=json.loads(row['matched_key_points'])
            )

        # 查询所有评测结果
        query = """
            SELECT r.id                          AS qa_id,
                q.question                    AS split_question,
                q.source_question_id          AS source_question_id,
                q.source_ml_id                AS source_ml_id,
                a.answer                      AS split_answer,
                a.key_points                  AS key_points,
                a.status_id                   AS status_id,
                mea.id                        AS model_answer_id,
                mea.model_id                  AS llm_model_id,
                m.model_name                  AS model_name,
                m.model_id                    AS model_id,
                m.model_remark                AS model_remark,
                m.release_date                AS model_release_date,
                mea.reasoning_content         AS reasoning_content,
                mea.answer_content            AS answer_content,
                aer.id                        AS eval_id,
                aer.missed_key_points         AS missed_key_points,
                aer.partial_key_points        AS partial_key_points,
                aer.matched_key_points        AS matched_key_points,
                aer.factual_errors_merged     AS factual_errors_merged,
                aer.vague_statements_merged   AS vague_statements_merged,
                aer.partially_correct_merged  AS partially_correct_merged,
                aer.irrelevant_correct_merged AS irrelevant_correct_merged,
                aer.factual_errors_unmerged   AS factual_errors_unmerged,
                aer.vague_statements_unmerged AS vague_statements_unmerged,
                aer.partially_correct_unmerged      AS partially_correct_unmerged,
                aer.irrelevant_correct_unmerged     AS irrelevant_correct_unmerged,
                aer.factual_errors_unmatched   AS factual_errors_unmatched,
                aer.vague_statements_unmatched AS vague_statements_unmatched,
                aer.partially_correct_unmatched      AS partially_correct_unmatched,
                aer.irrelevant_correct_unmatched     AS irrelevant_correct_unmatched,
                aer.extra_key_points          AS extra_key_points,
                aer.missed_key_points_details AS missed_key_points_details,
                cl.level                      AS cognitive_level,
                cl.name                       AS cognitive_level_name,
                cl.description                AS cognitive_level_description,
                vsa.is_version_specific       AS is_version_specific,
                COALESCE(ml.first_email_date, oa.creation_date)         AS qa_date
            FROM aggregated_eval_results AS aer
                    JOIN model_eval_answers mea ON aer.model_answer_id = mea.id
                    JOIN llm_models m ON m.id = mea.model_id
                    JOIN standard_qa_relations r ON r.id = mea.qa_id
                    JOIN standard_questions q ON r.question_id = q.id
                    JOIN standard_answers a ON r.answer_id = a.id
                    JOIN standard_question_cognitive_levels sqcl ON q.id = sqcl.standard_question_id
                    JOIN cognitive_levels cl ON cl.id = sqcl.cognitive_level_id
                    JOIN version_specific_analysis vsa ON vsa.qa_relation_id = r.id
                    JOIN dataset_maps dm ON dm.qa_id = r.id
                    LEFT JOIN mailing_lists ml ON ml.id = a.source_ml_id
                    LEFT JOIN original_answers oa ON oa.id = a.source_answer_id
            WHERE dm.version_id = %s
        """
        if len(model_ids) != 0:
            placeholders = ', '.join(['%s'] * len(model_ids))
            query += f'AND m.id IN ({placeholders})'
        if dataset_source != 0:
            if dataset_source == 1:
                query += 'AND q.source_question_id IS NOT NULL'
            elif dataset_source == 2:
                query += 'AND q.source_ml_id IS NOT NULL'
        
        # print(f"query_sql:\n{query}")
        cursor.execute(query, [dataset_version] + model_ids)
        rows = cursor.fetchall()

        evalResultList = []

        for row in rows:
            # 对每个评测结果获取其所有主题分类
            topics_query = """
                SELECT tc.id as topic_id, 
                       tc.name as topic_name, 
                       tc.description as topic_description
                FROM standard_questions q
                    JOIN standard_qa_relations r ON r.question_id = q.id
                    JOIN model_eval_answers mea ON mea.qa_id = r.id
                    JOIN aggregated_eval_results aer ON aer.model_answer_id = mea.id
                    JOIN standard_question_topics sqt ON q.id = sqt.standard_question_id
                    JOIN topic_categories tc ON sqt.topic_id = tc.id
                WHERE aer.id = %s
            """
            cursor.execute(topics_query, (row['eval_id'],))
            topics = [
                TopicInfo.from_db(topic_row)
                for topic_row in cursor.fetchall()
            ]

            model_answer_id = row['model_answer_id']
            judge_evals = judge_evals_map.get(model_answer_id, {})
            eval_result = EvalResult.from_db(row, topics, judge_evals)
            evalResultList.append(eval_result)

        return evalResultList

    finally:
        cursor.close()
        conn.close()

def get_EvalResult_by_qa_id(qa_id: int, model_ids: List[int] = [], dataset_version: int = 3, dataset_source: int = 0) -> List[EvalResult]:
    """
    从数据库获取指定问答的测评结果
    
    参数：
        qa_id：指定问答的id

        model_ids：指定被测模型的结果（默认为全部）  

        dataset_version：指定数据集版本（默认为3，最新）  

        dataset_source：指定问答对来源，0:全部，1:来自QA，2:来自邮件
    """
    # 创建数据库连接
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor(dictionary=True)

    try:

        # 先查 model_eval_results 表，构建 judge_evals 映射
        judge_query = """
            SELECT model_answer_id,
                   judge_model_id,
                   missed_key_points,
                   partial_key_points,
                   matched_key_points
            FROM model_eval_results
        """
        cursor.execute(judge_query)
        judge_rows = cursor.fetchall()

        from collections import defaultdict
        import json

        judge_evals_map: Dict[int, Dict[int, JudgeEvalResult]] = defaultdict(dict)

        for row in judge_rows:
            model_answer_id = row['model_answer_id']
            judge_model_id = row['judge_model_id']
            judge_evals_map[model_answer_id][judge_model_id] = JudgeEvalResult(
                missed_key_points=json.loads(row['missed_key_points']),
                partial_key_points=json.loads(row['partial_key_points']),
                matched_key_points=json.loads(row['matched_key_points'])
            )

        # 查询所有评测结果
        query = """
            SELECT r.id                          AS qa_id,
                q.question                    AS split_question,
                q.source_question_id          AS source_question_id,
                q.source_ml_id                AS source_ml_id,
                a.answer                      AS split_answer,
                a.key_points                  AS key_points,
                a.status_id                   AS status_id,
                mea.id                        AS model_answer_id,
                mea.model_id                  AS llm_model_id,
                m.model_name                  AS model_name,
                m.model_id                    AS model_id,
                m.model_remark                AS model_remark,
                m.release_date                AS model_release_date,
                mea.reasoning_content         AS reasoning_content,
                mea.answer_content            AS answer_content,
                aer.id                        AS eval_id,
                aer.missed_key_points         AS missed_key_points,
                aer.partial_key_points        AS partial_key_points,
                aer.matched_key_points        AS matched_key_points,
                aer.factual_errors_merged     AS factual_errors_merged,
                aer.vague_statements_merged   AS vague_statements_merged,
                aer.partially_correct_merged  AS partially_correct_merged,
                aer.irrelevant_correct_merged AS irrelevant_correct_merged,
                aer.factual_errors_unmerged   AS factual_errors_unmerged,
                aer.vague_statements_unmerged AS vague_statements_unmerged,
                aer.partially_correct_unmerged      AS partially_correct_unmerged,
                aer.irrelevant_correct_unmerged     AS irrelevant_correct_unmerged,
                aer.factual_errors_unmatched   AS factual_errors_unmatched,
                aer.vague_statements_unmatched AS vague_statements_unmatched,
                aer.partially_correct_unmatched      AS partially_correct_unmatched,
                aer.irrelevant_correct_unmatched     AS irrelevant_correct_unmatched,
                aer.extra_key_points          AS extra_key_points,
                aer.missed_key_points_details AS missed_key_points_details,
                cl.level                      AS cognitive_level,
                cl.name                       AS cognitive_level_name,
                cl.description                AS cognitive_level_description,
                vsa.is_version_specific       AS is_version_specific,
                COALESCE(ml.first_email_date, oa.creation_date)         AS qa_date
            FROM aggregated_eval_results AS aer
                    JOIN model_eval_answers mea ON aer.model_answer_id = mea.id
                    JOIN llm_models m ON m.id = mea.model_id
                    JOIN standard_qa_relations r ON r.id = mea.qa_id
                    JOIN standard_questions q ON r.question_id = q.id
                    JOIN standard_answers a ON r.answer_id = a.id
                    JOIN standard_question_cognitive_levels sqcl ON q.id = sqcl.standard_question_id
                    JOIN cognitive_levels cl ON cl.id = sqcl.cognitive_level_id
                    JOIN version_specific_analysis vsa ON vsa.qa_relation_id = r.id
                    JOIN dataset_maps dm ON dm.qa_id = r.id
                    LEFT JOIN mailing_lists ml ON ml.id = a.source_ml_id
                    LEFT JOIN original_answers oa ON oa.id = a.source_answer_id
            WHERE r.id = %s AND dm.version_id = %s
        """
        if len(model_ids) != 0:
            placeholders = ', '.join(['%s'] * len(model_ids))
            query += f'AND m.id IN ({placeholders})'
        if dataset_source != 0:
            if dataset_source == 1:
                query += 'AND q.source_question_id IS NOT NULL'
            elif dataset_source == 2:
                query += 'AND q.source_ml_id IS NOT NULL'
        
        # print(f"query_sql:\n{query}")
        cursor.execute(query, [qa_id, dataset_version] + model_ids)
        rows = cursor.fetchall()

        evalResultList = []

        for row in rows:
            # 对每个评测结果获取其所有主题分类
            topics_query = """
                SELECT tc.id as topic_id, 
                       tc.name as topic_name, 
                       tc.description as topic_description
                FROM standard_questions q
                    JOIN standard_qa_relations r ON r.question_id = q.id
                    JOIN model_eval_answers mea ON mea.qa_id = r.id
                    JOIN aggregated_eval_results aer ON aer.model_answer_id = mea.id
                    JOIN standard_question_topics sqt ON q.id = sqt.standard_question_id
                    JOIN topic_categories tc ON sqt.topic_id = tc.id
                WHERE aer.id = %s
            """
            cursor.execute(topics_query, (row['eval_id'],))
            topics = [
                TopicInfo.from_db(topic_row)
                for topic_row in cursor.fetchall()
            ]

            model_answer_id = row['model_answer_id']
            judge_evals = judge_evals_map.get(model_answer_id, {})
            eval_result = EvalResult.from_db(row, topics, judge_evals)
            evalResultList.append(eval_result)

        return evalResultList

    finally:
        cursor.close()
        conn.close()