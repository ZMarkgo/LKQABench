from dataclasses import dataclass
from typing import List, Dict
import json
from enum import Enum

class QAStatusEnum(Enum):
    """问答对状态枚举类"""
    PENDING = 0
    ACTIVE = 1
    MODIFIED_ACTIVE = 2
    DELETED = 3
    PENDING_ACTIVE = 4
    MANUAL_ADDED = 5

@dataclass
class QAStatus:
    """standard_qa_status 表的实体类"""
    id: int
    name: str
    description: str

    @staticmethod
    def get_all_statuses(db_config):
        """
        获取所有QA状态列表
        
        Args:
            db_config: 数据库配置
            
        Returns:
            List[QAStatus]: QA状态对象列表
        """
        import mysql.connector
        
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)
        try:
            query = "SELECT id, name, description FROM standard_qa_status ORDER BY id"
            cursor.execute(query)
            results = cursor.fetchall()
            return [QAStatus(id=row['id'], name=row['name'], description=row['description']) for row in results]
        finally:
            cursor.close()
            conn.close()

ACTIVE_STATUS_ID = QAStatusEnum.ACTIVE.value
MODIFIED_ACTIVE_STATUS_ID = QAStatusEnum.MODIFIED_ACTIVE.value
DELETED_STATUS_ID = QAStatusEnum.DELETED.value
MANUAL_ADDED_STATUS_ID = QAStatusEnum.MANUAL_ADDED.value

@dataclass
class QAVersion:
    """dataset_versions 表的实体类"""
    id: int
    version: str
    description: str

    @staticmethod
    def get_all_versions(db_config) -> List["QAVersion"]:
        """
        获取所有数据集版本列表

        Args:
            db_config: 数据库配置 (dict)

        Returns:
            List[QAVersion]: 数据集版本对象列表
        """
        import mysql.connector

        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)
        try:
            query = "SELECT id, version, description FROM dataset_versions ORDER BY id"
            cursor.execute(query)
            results = cursor.fetchall()
            return [
                QAVersion(id=row["id"], version=row["version"], description=row["description"])
                for row in results
            ]
        finally:
            cursor.close()
            conn.close()

@dataclass
class TopicInfo:
    id: int # 关联记录id
    topic_id: int # 主题分类本身的id
    name: str
    description: str
    status_id: int
    status_reason: str

    @classmethod
    def from_db_row(cls, db_row: dict):
        return cls(
            id=db_row['id'],
            topic_id=db_row['topic_id'],
            name=db_row['name'],
            description=db_row['description'],
            status_id=db_row['status_id'],
            status_reason=db_row['status_reason']
        )

# Cognitive Dimension 认知深度分类
@dataclass
class CognitiveDimension:
    id: int # 关联记录id
    cognitive_level_id: int # 认知维度本身的id
    level: int
    name: str
    description: str
    status_id: int
    status_reason: str

    @classmethod
    def from_db_row(cls, db_row: dict):
        return cls(
            id=db_row['id'],
            cognitive_level_id=db_row['cognitive_level_id'],
            level=db_row['level'],
            name=db_row['name'],
            description=db_row['description'],
            status_id=db_row['status_id'],
            status_reason=db_row['status_reason']
        )

# 版本相关性分类
@dataclass
class VersionRelevance:
    id:int # 关联记录id
    is_version_specific: bool
    related_version: str
    version_specific_reason: str
    status_id: int
    status_reason: str

    @classmethod
    def from_db_row(cls, db_row: dict):
        return cls(
            id=db_row['id'],
            is_version_specific=db_row['is_version_specific'],
            related_version=db_row['related_version'],
            version_specific_reason=db_row['version_specific_reason'],
            status_id=db_row['status_id'],
            status_reason=db_row['status_reason']
        )

@dataclass
class StandardQA:
    """实体类: 标准问答对"""
    id: int
    source_question_id: int
    source_answer_id: int
    split_question: str
    split_answer: str
    key_points: List[str]
    
    q_status_id: int
    q_status_reason: str
    a_status_id: int
    a_status_reason: str


    @classmethod
    def from_db_row(cls, db_row):
        """将数据库行转换为 类 实例"""
        return cls(
            id=db_row['id'],
            source_question_id=db_row['source_question_id'],
            source_answer_id=db_row['source_answer_id'],
            split_question=db_row['question'],
            split_answer=db_row['answer'],
            key_points=json.loads(db_row['key_points']),  # 解析 JSON
            q_status_id=db_row['q_status_id'],
            q_status_reason=db_row['q_status_reason'],
            a_status_id=db_row['a_status_id'],
            a_status_reason=db_row['a_status_reason']
        )

QA_ACTIVE_STATUS_IDS = [QAStatusEnum.ACTIVE.value, QAStatusEnum.MODIFIED_ACTIVE.value]
# 用于SQL查询的判断语句：判断问对是否被激活
QA_ACTIVE_STATUS_STATEMENT = f"status_id in ({', '.join(map(str, QA_ACTIVE_STATUS_IDS))})"

@dataclass
class ModelAnswer:
    """qa_model_answers表的实体类"""
    id: int
    qa_id: int
    # 模型在数据库中的ID
    llm_model_id: int
    model_name: str
    # 模型的代号ID
    model_id: str
    model_remark: str
    reasoning_content: str
    answer_content: str
    updated_at: str
    created_at: str
    # 是否是新获取的回答（不来自于数据库）
    is_new: bool = True
    def get_answer(self):
        """获取答案"""
        return self.answer_content

    def get_reasoning(self):
        """获取推理过程"""
        return self.reasoning_content

    def format_simple_info(self):
        """格式化简单信息：展示哪个模型回答了哪个问题"""
        reasoning_part = "无模型思考过程" if not self.reasoning_content else f"【模型思考过程】\n{self.reasoning_content}"
        answer_part = "无模型回答" if not self.answer_content else f"【模型回答】\n{self.answer_content}"
        return f"模型{self.model_name}({self.model_id}), qa({self.qa_id})\n{reasoning_part}\n{answer_part}"

    @classmethod
    def from_model_answer(cls, 
                          qa_id, 
                          llm_model_id, model_name, model_id, model_remark, 
                          answer_content:str, reasoning_content:str):
        """从模型答案创建类实例"""
        return cls(
            id=None,
            qa_id=qa_id,
            llm_model_id=llm_model_id,
            model_name=model_name,
            model_id=model_id,
            model_remark=model_remark,
            reasoning_content=reasoning_content,
            answer_content=answer_content,
            updated_at=None,
            created_at=None,
            is_new=True
        )

    @classmethod
    def from_db_row(cls, db_row:dict):
        """将数据库行转换为类实例；db_row中没有的键值对会被设置为None"""
        return cls(
            id=db_row.get('id'),
            qa_id=db_row.get('qa_id'),
            llm_model_id=db_row.get('llm_model_id'),
            model_name=db_row.get('model_name'),
            model_id=db_row.get('model_id'),
            model_remark=db_row.get('model_remark'),
            reasoning_content=db_row.get('reasoning_content'),
            answer_content=db_row.get('answer_content'),
            updated_at=db_row.get('updated_at', None),
            created_at=db_row.get('created_at', None),
            is_new=False
        )

@dataclass
class ModelConfigSimple:
    """llm_models表的实体类"""
    id: int
    model_name: str
    model_id: str
    model_remark: str
    api_type: str
    model_type: str

    @classmethod
    def from_db_row(cls, db_row: dict):
        """将数据库行转换为类实例"""
        return cls(
            id=db_row.get('id'),
            model_name=db_row.get('model_name'),
            model_id=db_row.get('model_id'),
            model_remark=db_row.get('model_remark'),
            api_type=db_row.get('api_type'),
            model_type=db_row.get('model_type')
        )

class IJudgeResult:
    """裁判结果的接口类, 返回的都是原始的字符串"""
    def get_model_answer_id(self) -> int:
        raise NotImplementedError("get_model_answer_id方法未实现")
    def get_judge_model_id(self) -> int:
        raise NotImplementedError("get_judge_model_id方法未实现")

    def get_missed_key_points(self) -> str:
        raise NotImplementedError("get_missed_key_points方法未实现")
    def get_partial_key_points(self) -> str:
        raise NotImplementedError("get_partial_key_points方法未实现")
    def get_matched_key_points(self) -> str:
        raise NotImplementedError("get_matched_key_points方法未实现")
   
    def get_factual_errors(self) -> str:
        raise NotImplementedError("get_factual_errors方法未实现")
    def get_partially_correct_but_misleading(self) -> str:
        raise NotImplementedError("get_partially_correct_but_misleading方法未实现")
    
    def get_vague_statements(self) -> str:
        raise NotImplementedError("get_vague_statements方法未实现")
    def get_irrelevant_but_correct(self) -> str:
        raise NotImplementedError("get_irrelevant_but_correct方法未实现")
    
    def get_reasoning_content(self) -> str:
        raise NotImplementedError("get_reasoning_content方法未实现")
    def get_content(self) -> str:
        raise NotImplementedError("get_content方法未实现")
    def get_is_eval_error(self) -> bool:
        raise NotImplementedError("get_is_eval_error方法未实现")
    

@dataclass
class ParsedModelEvalResult:    
    """解析后的裁判结果
    missed_key_points: [3, 4]
    partial_key_points: [2]
    matched_key_points: [1]
    factual_errors: [
        {
        "exact_text": "CFS allocates CPU time using time slices",
        "explanation": "CFS does not use time slices, it uses vruntime for fair scheduling"
        }
    ]
    vague_statements: [ 
        {
        "exact_text": "Linux employs a certain data structure to manage processes",
        "explanation": "The statement is vague as it does not specify which data structure (red-black tree) is used"
        }
    ]
    partially_correct_but_misleading: [
        {
        "exact_text": "Linux supports BPF for flexible scheduling",
        "explanation": "While BPF can be used for scheduling, it is not directly related to the core scheduling mechanisms being discussed"
        }
    ]
    irrelevant_but_correct: [
        {
        "exact_text": "Linux supports BPF for flexible scheduling",
        "explanation": "While BPF can be used for scheduling, it is not directly related to the core scheduling mechanisms being discussed"
        }
    ]
    """
    id: int
    model_answer_id: int 
    judgeModelConfig: ModelConfigSimple

    missed_key_points: List[int]
    partial_key_points: List[int]
    matched_key_points: List[int]

    factual_errors: List[Dict[str, str]]
    vague_statements: List[Dict[str, str]]
    partially_correct_but_misleading: List[Dict[str, str]]
    irrelevant_but_correct: List[Dict[str, str]]

    reasoning_content: str
    content: str
    is_eval_error: bool

    updated_at: str
    created_at: str

    def to_dict(self) -> dict:
        """将对象转换为可序列化的字典"""
        return {
            'id': self.id,
            'model_answer_id': self.model_answer_id,
            'judge_model_config': {
                'id': self.judgeModelConfig.id,
                'model_name': self.judgeModelConfig.model_name,
                'model_id': self.judgeModelConfig.model_id,
                'model_remark': self.judgeModelConfig.model_remark,
                'api_type': self.judgeModelConfig.api_type,
                'model_type': self.judgeModelConfig.model_type
            },
            'missed_key_points': self.missed_key_points,
            'partial_key_points': self.partial_key_points,
            'matched_key_points': self.matched_key_points,
            'factual_errors': self.factual_errors,
            'vague_statements': self.vague_statements,
            'partially_correct_but_misleading': self.partially_correct_but_misleading,
            'irrelevant_but_correct': self.irrelevant_but_correct,
            'reasoning_content': self.reasoning_content,
            'content': self.content,
            'is_eval_error': self.is_eval_error,
            'updated_at': str(self.updated_at) if self.updated_at else None,
            'created_at': str(self.created_at) if self.created_at else None
        }

    def to_json(self, indent: int = None) -> str:
        """将对象转换为JSON字符串
        Args:
            indent: JSON缩进，默认为None（不缩进）
        Returns:
            str: JSON字符串
        """
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def list_to_json(cls, results: List['ParsedModelEvalResult'], indent: int = None) -> str:
        """将结果列表转换为JSON字符串
        Args:
            results: ParsedModelEvalResult对象列表
            indent: JSON缩进，默认为None（不缩进）
        Returns:
            str: JSON字符串
        """
        return json.dumps([result.to_dict() for result in results], indent=indent, ensure_ascii=False)

    @classmethod
    def from_db_row(cls, judge_result_db_row: dict, judge_model_config_db_row: dict):
        """将数据库行转换为类实例"""
        return cls(
            id=judge_result_db_row.get('id'),
            model_answer_id=judge_result_db_row.get('model_answer_id'),
            judgeModelConfig=ModelConfigSimple.from_db_row(judge_model_config_db_row),

            missed_key_points=json.loads(judge_result_db_row.get('missed_key_points', '[]')),
            partial_key_points=json.loads(judge_result_db_row.get('partial_key_points', '[]')),
            matched_key_points=json.loads(judge_result_db_row.get('matched_key_points', '[]')),

            factual_errors=json.loads(judge_result_db_row.get('factual_errors', '[]')),
            vague_statements=json.loads(judge_result_db_row.get('vague_statements', '[]')),
            partially_correct_but_misleading=json.loads(judge_result_db_row.get('partially_correct_but_misleading', '[]')),
            irrelevant_but_correct=json.loads(judge_result_db_row.get('irrelevant_but_correct', '[]')),
            
            reasoning_content=judge_result_db_row.get('reasoning_content'),
            content=judge_result_db_row.get('content'),
            is_eval_error=judge_result_db_row.get('is_eval_error'),
            
            updated_at=judge_result_db_row.get('updated_at'),
            created_at=judge_result_db_row.get('created_at')
        )

    @classmethod
    def from_judge_result(cls, judgeResult: IJudgeResult, judge_model_config_db_row: dict):
        """从裁判结果和裁判模型配置创建类实例"""
        return cls(
            id=None,
            model_answer_id=judgeResult.get_model_answer_id(),
            judgeModelConfig=ModelConfigSimple.from_db_row(judge_model_config_db_row),

            missed_key_points=judgeResult.get_missed_key_points() or [],
            partial_key_points=judgeResult.get_partial_key_points() or [],
            matched_key_points=judgeResult.get_matched_key_points() or [],

            factual_errors=judgeResult.get_factual_errors() or [],
            vague_statements=judgeResult.get_vague_statements() or [],
            partially_correct_but_misleading=judgeResult.get_partially_correct_but_misleading() or [],
            irrelevant_but_correct=judgeResult.get_irrelevant_but_correct() or [],
            reasoning_content=judgeResult.get_reasoning_content(),
            content=judgeResult.get_content(),
            is_eval_error=judgeResult.get_is_eval_error(),

            updated_at=None,
            created_at=None
        )

    def format_judge_model_info(self):
        """格式化的裁判模型信息"""
        return f"{self.judgeModelConfig.model_name}(id={self.judgeModelConfig.id})"

