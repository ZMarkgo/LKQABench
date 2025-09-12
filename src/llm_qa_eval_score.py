import mysql.connector
import os
import json
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from common.system import ErrorCounter
from common.timer import timer
from common.logger import setup_logger, get_logger, TimestampType
from common.config import load_config, get_db_config, get_api_key_secret_key
from common.llm_api import ModelConfig, load_models_configs
from common.api_key_encryptor import ApiKeyEncryptor
from common.eval_result import get_EvalResult, get_EvalResult_by_qa_id, EvalResult
import argparse
from dataclasses import dataclass, asdict
from datetime import datetime

# 读取配置
CONFIG_NAME = "llm_qa_eval_score"
CONFIG = load_config()
DB_CONFIG = get_db_config(CONFIG)
LOG_FILE = CONFIG[CONFIG_NAME]['log_file']

if __name__ == "__main__":
    # 将在main函数中配置
    logger = None  
else:
    # 当作为模块导入时，使用默认配置
    logger = get_logger(__name__)

@dataclass
class ScoreParams:
    """评分参数类"""
    # 问答对来源
    qa_src: str
    # 问答对ID列表
    qa_id_list: List[int]
    # 模型ID列表
    model_id_list: List[int]
    # 数据集版本
    dataset_version: int
    # 数据集来源
    dataset_source: int
    # 输出文件路径
    output_file: str
    # 评分方法配置
    coverage_weight: float
    accuracy_weight: float
    clarity_weight: float
    factual_error_penalty: float
    misleading_penalty: float
    vague_penalty: float
    irrelevant_penalty: float
    coverage_score_method: int
    unmerged_errors_valid: bool
    accuracy_and_clarity_positive: bool

def get_available_models() -> List[Dict[str, Any]]:
    """
    获取数据库中所有可用的模型列表
    
    Returns:
        List[Dict[str, Any]]: 模型信息列表
    """
    conn = mysql.connector.connect(**DB_CONFIG)
    try:
        with conn.cursor(dictionary=True) as cursor:
            cursor.execute("""
                SELECT id, model_name, model_id, model_remark
                FROM llm_models
                ORDER BY id
            """)
            return cursor.fetchall()
    finally:
        conn.close()

def get_available_qa_relations(dataset_version: int = 3, limit: int = 100) -> List[Dict[str, Any]]:
    """
    获取数据库中可用的问答对关系列表
    
    Args:
        dataset_version: 数据集版本
        limit: 限制返回数量
        
    Returns:
        List[Dict[str, Any]]: 问答对关系信息列表
    """
    conn = mysql.connector.connect(**DB_CONFIG)
    try:
        with conn.cursor(dictionary=True) as cursor:
            cursor.execute("""
                SELECT r.id as qa_id, 
                       q.question as split_question,
                       a.answer as split_answer,
                       q.source_question_id,
                       q.source_ml_id
                FROM standard_qa_relations r
                JOIN standard_questions q ON r.question_id = q.id
                JOIN standard_answers a ON r.answer_id = a.id
                JOIN dataset_maps dm ON dm.qa_id = r.id
                WHERE dm.version_id = %s
                AND ((q.status_id = 1 or q.status_id = 2 or q.status_id = 4) AND
                     (a.status_id = 1 or a.status_id = 2 or a.status_id = 4))
                ORDER BY r.id
                LIMIT %s
            """, (dataset_version, limit))
            return cursor.fetchall()
    finally:
        conn.close()

def calculate_scores_for_models(params: ScoreParams) -> Dict[str, Any]:
    """
    为指定的模型和问答对计算评分
    
    Args:
        params (ScoreParams): 评分参数
        
    Returns:
        Dict[str, Any]: 包含评分结果的字典
    """
    logger.info(f"开始计算评分，问答对来源: {params.qa_src}")
    
    # 获取评测结果
    eval_results = []
    
    if params.qa_src == 'all':
        # 获取所有评测结果
        eval_results = get_EvalResult(
            model_ids=params.model_id_list,
            dataset_version=params.dataset_version,
            dataset_source=params.dataset_source
        )
        logger.info(f"获取到 {len(eval_results)} 个评测结果")
    elif params.qa_src == 'list':
        # 根据问答对ID列表获取评测结果
        for qa_id in params.qa_id_list:
            qa_eval_results = get_EvalResult_by_qa_id(
                qa_id=qa_id,
                model_ids=params.model_id_list,
                dataset_version=params.dataset_version,
                dataset_source=params.dataset_source
            )
            eval_results.extend(qa_eval_results)
        logger.info(f"获取到 {len(eval_results)} 个评测结果")
    
    if not eval_results:
        logger.warning("没有找到符合条件的评测结果")
        return {
            "timestamp": datetime.now().isoformat(),
            "parameters": asdict(params),
            "summary": {
                "total_evaluations": 0,
                "models_count": 0,
                "qa_pairs_count": 0
            },
            "results": []
        }
    
    # 重新计算评分
    for eval_result in eval_results:
        eval_result.calculate_scores(
            coverage_weight=params.coverage_weight,
            accuracy_weight=params.accuracy_weight,
            clarity_weight=params.clarity_weight,
            factual_error_penalty=params.factual_error_penalty,
            misleading_penalty=params.misleading_penalty,
            vague_penalty=params.vague_penalty,
            irrelevant_penalty=params.irrelevant_penalty,
            coverage_score_method=params.coverage_score_method,
            unmerged_errors_valid=params.unmerged_errors_valid,
            accuracy_and_clarity_positive=params.accuracy_and_clarity_positive
        )
    
    # 按模型分组统计
    model_stats = {}
    qa_ids = set()
    
    for eval_result in eval_results:
        model_key = f"{eval_result.model_name}_{eval_result.model_id}"
        qa_ids.add(eval_result.qa_id)
        
        if model_key not in model_stats:
            model_stats[model_key] = {
                "model_info": {
                    "llm_model_id": eval_result.llm_model_id,
                    "model_name": eval_result.model_name,
                    "model_id": eval_result.model_id,
                    "model_remark": eval_result.model_remark
                },
                "evaluations": [],
                "statistics": {
                    "count": 0,
                    "avg_total_score": 0.0,
                    "avg_coverage_score": 0.0,
                    "avg_accuracy_score": 0.0,
                    "avg_clarity_score": 0.0,
                    "total_factual_errors": 0,
                    "total_misleading": 0,
                    "total_vague_statements": 0,
                    "total_irrelevant": 0
                }
            }
        
        # 添加单个评测结果
        evaluation = {
            "qa_id": eval_result.qa_id,
            "split_question": eval_result.split_question,
            "answer_content": eval_result.answer_content,
            "scores": {
                "total_score": eval_result.total_score,
                "coverage_score": eval_result.coverage_score,
                "accuracy_score": eval_result.accuracy_score,
                "clarity_score": eval_result.clarity_score,
                "raw_accuracy_score": eval_result.raw_accuracy_score,
                "raw_clarity_score": eval_result.raw_clarity_score
            },
            "score_breakdown": eval_result.get_score_breakdown(),
            "topics": [{
                "topic_id": topic.topic_id,
                "topic_name": topic.topic_name,
                "topic_description": topic.topic_description
            } for topic in eval_result.topics],
            "cognitive_level": {
                "level": eval_result.cognitive_level,
                "name": eval_result.cognitive_level_name,
                "description": eval_result.cognitive_level_description
            },
            "is_version_specific": eval_result.is_version_specific,
            "qa_date": eval_result.qa_date.isoformat() if eval_result.qa_date else None,
            "model_release_date": eval_result.model_release_date.isoformat() if eval_result.model_release_date else None
        }
        
        model_stats[model_key]["evaluations"].append(evaluation)
        
        # 更新统计信息
        stats = model_stats[model_key]["statistics"]
        stats["count"] += 1
        stats["total_factual_errors"] += len(eval_result.factual_errors)
        stats["total_misleading"] += len(eval_result.partially_correct_but_misleading)
        stats["total_vague_statements"] += len(eval_result.vague_statements)
        stats["total_irrelevant"] += len(eval_result.irrelevant_but_correct)
    
    # 计算平均分
    for model_key, model_data in model_stats.items():
        stats = model_data["statistics"]
        count = stats["count"]
        if count > 0:
            total_scores = [eval["scores"]["total_score"] for eval in model_data["evaluations"]]
            coverage_scores = [eval["scores"]["coverage_score"] for eval in model_data["evaluations"]]
            accuracy_scores = [eval["scores"]["accuracy_score"] for eval in model_data["evaluations"]]
            clarity_scores = [eval["scores"]["clarity_score"] for eval in model_data["evaluations"]]
            
            stats["avg_total_score"] = sum(total_scores) / count
            stats["avg_coverage_score"] = sum(coverage_scores) / count
            stats["avg_accuracy_score"] = sum(accuracy_scores) / count
            stats["avg_clarity_score"] = sum(clarity_scores) / count
    
    # 构建最终结果
    result = {
        "timestamp": datetime.now().isoformat(),
        "parameters": asdict(params),
        "summary": {
            "total_evaluations": len(eval_results),
            "models_count": len(model_stats),
            "qa_pairs_count": len(qa_ids)
        },
        "results": list(model_stats.values())
    }
    
    return result

def save_results_to_file(results: Dict[str, Any], output_file: str):
    """
    将结果保存到JSON文件
    
    Args:
        results: 评分结果字典
        output_file: 输出文件路径
    """
    abs_output_file = os.path.abspath(output_file)
    
    # 确保输出目录存在
    output_dir = os.path.dirname(abs_output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(abs_output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"评分结果已保存到: {abs_output_file}")

def get_config_value(key, default_value, value_type=str):
    """
    从配置文件读取参数的辅助函数
    
    Args:
        key: 配置键名
        default_value: 默认值
        value_type: 值类型
        
    Returns:
        配置值或默认值
    """
    try:
        if CONFIG_NAME in CONFIG and key in CONFIG[CONFIG_NAME]:
            value = CONFIG[CONFIG_NAME][key]
            if value_type == bool:
                return bool(value)
            elif value_type == int:
                return int(value)
            elif value_type == float:
                return float(value)
            elif value_type == list:
                return list(value) if isinstance(value, list) else []
            else:
                return str(value)
        return default_value
    except (ValueError, TypeError):
        return default_value

def parse_args():
    """
    解析命令行参数
    
    Returns:
        argparse.Namespace: 解析后的参数对象
    """
    parser = argparse.ArgumentParser(
        description='模型评分计算工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  1. 计算所有问答对和模型的评分:
     python llm_qa_eval_score.py --qa-src all --output-file results.json
  
  2. 计算指定问答对的评分:
     python llm_qa_eval_score.py --qa-src list --qa-id-list 1,2,3 --output-file results.json
  
  3. 计算指定模型的评分:
     python llm_qa_eval_score.py --qa-src all --model-id-list 1,2 --output-file results.json
  
  4. 自定义评分权重:
     python llm_qa_eval_score.py --qa-src all --coverage-weight 60 --accuracy-weight 25 --clarity-weight 15
  
  5. 启用debug模式:
     python llm_qa_eval_score.py --debug --qa-src list --qa-id-list 1
  
  6. 列出可用的模型:
     python llm_qa_eval_score.py --list-models
  
  7. 列出可用的问答对:
     python llm_qa_eval_score.py --list-qa-pairs
"""
    )
    
    # 从配置文件读取参数
    DEFAULT_QA_SRC = get_config_value('qa_src', 'all')
    DEFAULT_DATASET_VERSION = get_config_value('dataset_version', 3, int)
    DEFAULT_DATASET_SOURCE = get_config_value('dataset_source', 0, int)
    DEFAULT_OUTPUT_FILE = get_config_value('output_file', 'eval_scores.json')
    DEFAULT_QA_ID_LIST = get_config_value('qa_id_list', [], list)
    DEFAULT_MODEL_ID_LIST = get_config_value('model_id_list', [], list)
    
    # 问答对来源
    parser.add_argument(
        '--qa-src',
        type=str,
        choices=['all', 'list'],
        default=DEFAULT_QA_SRC,
        help=f'问答对来源: all 表示选择所有；list 表示从qa-id-list作为问答来源 (默认: {DEFAULT_QA_SRC})'
    )
    
    # 问答对ID列表
    parser.add_argument(
        '--qa-id-list',
        type=str,
        help='问答对ID列表, 用逗号分隔, 仅在qa-src=list时有效'
    )
    
    # 模型ID列表
    parser.add_argument(
        '--model-id-list',
        type=str,
        help='模型ID列表, 用逗号分隔, 不指定则计算所有模型'
    )
    
    # 数据集版本
    parser.add_argument(
        '--dataset-version',
        type=int,
        default=DEFAULT_DATASET_VERSION,
        help=f'数据集版本 (默认: {DEFAULT_DATASET_VERSION})'
    )
    
    # 数据集来源
    parser.add_argument(
        '--dataset-source',
        type=int,
        choices=[0, 1, 2],
        default=DEFAULT_DATASET_SOURCE,
        help=f'数据集来源: 0=全部, 1=来自QA, 2=来自邮件 (默认: {DEFAULT_DATASET_SOURCE})'
    )
    
    # 输出文件
    parser.add_argument(
        '--output-file',
        type=str,
        default=DEFAULT_OUTPUT_FILE,
        help=f'输出JSON文件路径 (默认: {DEFAULT_OUTPUT_FILE})'
    )
    

    
    # 调试模式
    parser.add_argument(
        '--debug',
        action='store_true',
        help='启用调试模式（默认关闭），包括使用秒级时间戳记录日志'
    )
    
    # 列出可用资源
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='列出数据库中所有可用的模型'
    )
    
    parser.add_argument(
        '--list-qa-pairs',
        action='store_true',
        help='列出数据库中可用的问答对（限制前100个）'
    )
    
    return parser.parse_args()

@timer
def main():
    """
    主函数，负责解析命令行参数并执行评分计算任务
    """
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
        
        # 处理列出资源的请求
        if args.list_models:
            logger.info("列出所有可用模型:")
            models = get_available_models()
            for model in models:
                print(f"ID: {model['id']}, 名称: {model['model_name']}, 模型ID: {model['model_id']}, 备注: {model['model_remark']}")
            return
        
        if args.list_qa_pairs:
            logger.info("列出可用问答对（前100个）:")
            qa_pairs = get_available_qa_relations(dataset_version=args.dataset_version)
            for qa in qa_pairs:
                print(f"QA ID: {qa['qa_id']}, 问题: {qa['split_question'][:100]}...")
            return
        
        # 从配置文件读取评分参数
        coverage_weight = get_config_value('coverage_weight', 50.0, float)
        accuracy_weight = get_config_value('accuracy_weight', 30.0, float)
        clarity_weight = get_config_value('clarity_weight', 20.0, float)
        factual_error_penalty = get_config_value('factual_error_penalty', 15.0, float)
        misleading_penalty = get_config_value('misleading_penalty', 10.0, float)
        vague_penalty = get_config_value('vague_penalty', 8.0, float)
        irrelevant_penalty = get_config_value('irrelevant_penalty', 5.0, float)
        coverage_score_method = get_config_value('coverage_score_method', 2, int)
        unmerged_errors_valid = get_config_value('unmerged_errors_valid', False, bool)
        accuracy_and_clarity_positive = get_config_value('accuracy_and_clarity_positive', True, bool)
        
        # 验证权重总和
        total_weight = coverage_weight + accuracy_weight + clarity_weight
        if abs(total_weight - 100.0) > 0.01:
            raise ValueError(f"权重总和必须为100，当前为: {total_weight}")
        
        # 处理问答对来源参数
        qa_id_list = []
        if args.qa_src == 'list':
            if not args.qa_id_list:
                raise ValueError("使用 --qa-src list 时必须指定 --qa-id-list")
            qa_id_list = [int(id.strip()) for id in args.qa_id_list.split(',')]
        
        # 处理模型ID列表
        model_id_list = []
        if args.model_id_list:
            model_id_list = [int(id.strip()) for id in args.model_id_list.split(',')]
        
        # 输出运行参数
        logger.info("运行参数配置:")
        logger.info(f"  问答对来源(qa-src): {args.qa_src}")
        if args.qa_src == 'list':
            logger.info(f"  问答对ID列表(qa-id-list): {qa_id_list}")
        if model_id_list:
            logger.info(f"  模型ID列表(model-id-list): {model_id_list}")
        logger.info(f"  数据集版本(dataset-version): {args.dataset_version}")
        logger.info(f"  数据集来源(dataset-source): {args.dataset_source}")
        logger.info(f"  输出文件(output-file): {args.output_file}")
        logger.info(f"  评分权重 - 覆盖度: {coverage_weight}, 正确性: {accuracy_weight}, 清晰度: {clarity_weight}")
        logger.info(f"  惩罚参数 - 事实错误: {factual_error_penalty}, 误导: {misleading_penalty}, 模糊: {vague_penalty}, 无关: {irrelevant_penalty}")
        logger.info(f"  覆盖率评分方法: {coverage_score_method}")
        logger.info(f"  未融合错误有效: {unmerged_errors_valid}")
        logger.info(f"  正确性和清晰度保持正值: {accuracy_and_clarity_positive}")
        
        # 创建评分参数对象
        score_params = ScoreParams(
            qa_src=args.qa_src,
            qa_id_list=qa_id_list,
            model_id_list=model_id_list,
            dataset_version=args.dataset_version,
            dataset_source=args.dataset_source,
            output_file=args.output_file,
            coverage_weight=coverage_weight,
            accuracy_weight=accuracy_weight,
            clarity_weight=clarity_weight,
            factual_error_penalty=factual_error_penalty,
            misleading_penalty=misleading_penalty,
            vague_penalty=vague_penalty,
            irrelevant_penalty=irrelevant_penalty,
            coverage_score_method=coverage_score_method,
            unmerged_errors_valid=unmerged_errors_valid,
            accuracy_and_clarity_positive=accuracy_and_clarity_positive
        )
        
        # 计算评分
        logger.info("开始计算模型评分...")
        results = calculate_scores_for_models(score_params)
        
        # 保存结果
        save_results_to_file(results, args.output_file)
        
        # 输出摘要信息
        summary = results["summary"]
        logger.info(f"评分计算完成！")
        logger.info(f"  总评测数量: {summary['total_evaluations']}")
        logger.info(f"  模型数量: {summary['models_count']}")
        logger.info(f"  问答对数量: {summary['qa_pairs_count']}")
        
        # 输出各模型平均分
        for model_result in results["results"]:
            model_info = model_result["model_info"]
            stats = model_result["statistics"]
            logger.info(f"  模型 {model_info['model_name']} ({model_info['model_id']}) - 平均总分: {stats['avg_total_score']:.2f}")
        
    except Exception as e:
        logger.error(f"执行出错: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()