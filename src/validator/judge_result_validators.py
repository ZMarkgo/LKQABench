"""裁判结果验证模块

该模块提供了用于验证裁判模型评估结果的工具类和方法。
主要功能包括JSON解析、结构验证和内容完整性检查。
"""

import json
from typing import List, Tuple, Dict, Any
from common import json_tools
from common.logger import get_logger

logger = get_logger(__name__)


class JudgeResultValidator:
    """裁判结果JSON解析和验证工具类
    
    该类提供了一系列静态方法用于验证裁判模型输出的JSON结果，
    包括JSON解析、结构完整性验证和内容正确性检查。
    """
    
    @staticmethod
    def parse_judge_result_json(content: str) -> Tuple[bool, Dict[str, Any], str]:
        """解析裁判结果JSON内容。
        
        Args:
            content: JSON格式的裁判结果内容
            
        Returns:
            Tuple[bool, Dict[str, Any], str]: 包含是否解析成功、解析后的数据字典和错误信息的元组
        """
        try:
            evaluation_data = json_tools.json_load_llm_answer(content)
            return True, evaluation_data, ""
        except Exception as e:
            error_msg = f"JSON解析失败: {str(e)}"
            logger.debug(f"JSON解析失败，内容: {content[:200]}..., 错误: {error_msg}")
            return False, {}, error_msg
    
    @staticmethod
    def validate_judge_result_structure(evaluation_data: Dict[str, Any]) -> Tuple[bool, List[str], Dict[str, Any]]:
        """验证裁判结果的结构完整性和正确性。
        
        Args:
            evaluation_data: 解析后的评估数据字典
            
        Returns:
            Tuple[bool, List[str], Dict[str, Any]]: 包含是否通过验证、错误信息列表和统计信息字典的元组
        """
        stats = {
            'field_errors': {
                'key_points_evaluation': 0,
                'factual_errors': 0,
                'vague_statements': 0,
                'partially_correct_but_misleading': 0,
                'irrelevant_but_correct': 0
            },
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
        
        errors = []
        
        # 检查必要字段
        required_fields = [
            'key_points_evaluation',
            'factual_errors',
            'vague_statements',
            'partially_correct_but_misleading',
            'irrelevant_but_correct'
        ]
        
        missing_fields = []
        for field in required_fields:
            if field not in evaluation_data:
                missing_fields.append(field)
                stats['field_errors'][field] += 1
        
        if missing_fields:
            errors.append(f"缺少必要字段: {missing_fields}")
            return False, errors, stats
        
        # 检查key_points_evaluation字段结构
        kp_eval = evaluation_data['key_points_evaluation']
        if not isinstance(kp_eval, dict):
            stats['structure_errors']['key_points_evaluation']['not_dict'] += 1
            errors.append("key_points_evaluation 必须是字典类型")
            return False, errors, stats
            
        required_kp_fields = ['missed', 'partial', 'matched']
        for field in required_kp_fields:
            if field not in kp_eval:
                stats['structure_errors']['key_points_evaluation']['missing_fields'][field] += 1
                errors.append(f"key_points_evaluation 缺少必要字段: {field}")
                return False, errors, stats
            if not isinstance(kp_eval[field], list):
                stats['structure_errors']['key_points_evaluation']['not_list'][field] += 1
                errors.append(f"key_points_evaluation.{field} 必须是列表类型")
                return False, errors, stats
        
        # 检查其他字段结构
        statement_fields = [
            'factual_errors',
            'vague_statements',
            'partially_correct_but_misleading',
            'irrelevant_but_correct'
        ]
        
        for field in statement_fields:
            if not isinstance(evaluation_data[field], list):
                stats['structure_errors']['statement_fields']['not_list'] += 1
                errors.append(f"{field} 必须是列表类型")
                return False, errors, stats
            
            # 检查列表中的每个项目
            for item in evaluation_data[field]:
                if not isinstance(item, dict):
                    stats['structure_errors']['statement_fields']['item_not_dict'] += 1
                    errors.append(f"{field} 中的项目必须是字典类型")
                    return False, errors, stats
                if 'exact_text' not in item:
                    stats['structure_errors']['statement_fields']['missing_fields']['exact_text'] += 1
                    errors.append(f"{field} 中的项目缺少 exact_text 字段")
                    return False, errors, stats
                if 'explanation' not in item:
                    stats['structure_errors']['statement_fields']['missing_fields']['explanation'] += 1
                    errors.append(f"{field} 中的项目缺少 explanation 字段")
                    return False, errors, stats
                if not isinstance(item['exact_text'], str):
                    stats['structure_errors']['statement_fields']['not_string']['exact_text'] += 1
                    errors.append(f"{field} 中的 exact_text 必须是字符串类型")
                    return False, errors, stats
                if not isinstance(item['explanation'], str):
                    stats['structure_errors']['statement_fields']['not_string']['explanation'] += 1
                    errors.append(f"{field} 中的 explanation 必须是字符串类型")
                    return False, errors, stats
        
        return True, errors, stats
    
    @staticmethod
    def validate_judge_result_content(content: str) -> Tuple[bool, List[str], Dict[str, Any]]:
        """验证裁判结果内容的完整性和正确性。
        
        这是一个便捷方法，结合了JSON解析和结构验证。
        
        Args:
            content: JSON格式的裁判结果内容
            
        Returns:
            Tuple[bool, List[str], Dict[str, Any]]: 包含是否通过验证、错误信息列表和统计信息字典的元组
        """
        # 首先解析JSON
        parse_success, evaluation_data, parse_error = JudgeResultValidator.parse_judge_result_json(content)
        if not parse_success:
            return False, [parse_error], {}
        
        # 然后验证结构
        return JudgeResultValidator.validate_judge_result_structure(evaluation_data)
    
    @staticmethod
    def validate_content(content: str) -> Tuple[bool, List[str], Dict[str, Any]]:
        """validate_judge_result_content的简化别名。
        
        Args:
            content: JSON格式的裁判结果内容
            
        Returns:
            Tuple[bool, List[str], Dict[str, Any]]: 包含是否通过验证、错误信息列表和统计信息字典的元组
        """
        return JudgeResultValidator.validate_judge_result_content(content)
    
    @staticmethod
    def parse_json(content: str) -> Tuple[bool, Dict[str, Any], str]:
        """parse_judge_result_json的简化别名。
        
        Args:
            content: JSON格式的裁判结果内容
            
        Returns:
            Tuple[bool, Dict[str, Any], str]: 包含是否解析成功、解析后的数据字典和错误信息的元组
        """
        return JudgeResultValidator.parse_judge_result_json(content)


class JudgeResultValidationError(Exception):
    """裁判结果验证异常类
    
    用于表示裁判结果验证过程中发生的错误。
    """
    
    def __init__(self, message: str, errors: List[str] = None, stats: Dict[str, Any] = None):
        """初始化验证异常。
        
        Args:
            message: 错误消息
            errors: 详细错误列表
            stats: 统计信息
        """
        super().__init__(message)
        self.errors = errors or []
        self.stats = stats or {}
    
    def __str__(self):
        if self.errors:
            return f"{super().__str__()}: {'; '.join(self.errors)}"
        return super().__str__()


def validate_judge_result(content: str, raise_on_error: bool = False) -> Tuple[bool, List[str], Dict[str, Any]]:
    """验证裁判结果的便捷函数。
    
    Args:
        content: JSON格式的裁判结果内容
        raise_on_error: 是否在验证失败时抛出异常
        
    Returns:
        Tuple[bool, List[str], Dict[str, Any]]: 包含是否通过验证、错误信息列表和统计信息字典的元组
        
    Raises:
        JudgeResultValidationError: 当raise_on_error为True且验证失败时抛出
    """
    is_valid, errors, stats = JudgeResultValidator.validate_judge_result_content(content)
    
    if not is_valid and raise_on_error:
        raise JudgeResultValidationError("裁判结果验证失败", errors, stats)
    
    return is_valid, errors, stats