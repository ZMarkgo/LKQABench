import mysql.connector
import os
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from common.system import ErrorCounter
from common.timer import timer
from common.logger import setup_logger, get_logger, TimestampType
from common.config import load_config, get_db_config, get_api_key_secret_key
from common.entity import StandardQA, ModelAnswer
from common.llm_api import ModelConfig, load_models_configs
from common.llm_api import LLM
from common.api_key_encryptor import ApiKeyEncryptor
import argparse
from dataclasses import dataclass

# 读取配置
CONFIG_NAME = "llm_qa_eval_fetch"
CONFIG = load_config()
DB_CONFIG = get_db_config(CONFIG)
LOG_FILE = CONFIG[CONFIG_NAME]['log_file']
# 如果一个模型在获取回答时出错次数超过MAX_API_ERROR_LIMIT次, 则停止对该模型的评估, 不影响对其他模型的评估
MAX_API_ERROR_LIMIT = 3

if __name__ == "__main__":
    # 将在main函数中配置
    logger = None  
else:
    # 当作为模块导入时，使用默认配置
    logger = get_logger(__name__)

# 待测评大模型的系统提示词（System Prompt）
EVALUATED_SYSTEM_PROMPT = """
You are a Linux Kernel expert. Please provide a concise and professional answer in English.
"""

def get_unanswered_standard_qa_batch(offset, limit, llm_model_id) -> List[StandardQA]:
    """
    获取被测大模型还没有回答过的一批标准问答对
    
    Args:
        offset: 偏移量
        limit: 限制数量
        llm_model_id: 被测大模型的数据库ID
        
    Returns:
        List[StandardQA]: 一批标准问答对列表
    """
    conn = mysql.connector.connect(**DB_CONFIG)
    try:
        with conn.cursor(dictionary=True) as cursor:
            cursor.execute(f"""
                SELECT r.id                 AS id,
                    q.source_question_id AS source_question_id,
                    a.source_answer_id   AS source_answer_id,
                    q.question           as question,
                    a.answer             as answer,
                    a.key_points         as key_points,
                    q.status_id          as q_status_id,
                    q.status_reason      as q_status_reason,
                    a.status_id          as a_status_id,
                    a.status_reason      as a_status_reason
                FROM standard_qa_relations r
                        JOIN standard_questions q on r.question_id = q.id
                        JOIN standard_answers a on r.answer_id = a.id
                WHERE ((SELECT COUNT(*)
                        FROM model_eval_answers temp_mea
                        WHERE temp_mea.qa_id = r.id
                        AND temp_mea.model_id = %s) <= 0)
                AND ((q.status_id = 1 or q.status_id = 2 or q.status_id = 4) AND
                    (a.status_id = 1 or a.status_id = 2 or a.status_id = 4))
                LIMIT %s, %s
            """, (llm_model_id, offset, limit))
            return [StandardQA.from_db_row(row) for row in cursor.fetchall()]
    finally:
        conn.close()

def get_unanswered_standard_qa(qa_ids, offset, limit, llm_model_id) -> List[StandardQA]:
    """
    根据问答对ID列表获取被测大模型还没有回答过的一批标准问答对
    
    Args:
        qa_ids: 问答对ID列表
        offset: 偏移量
        limit: 限制数量
        llm_model_id: 被测大模型的数据库ID
        
    Returns:
        List[StandardQA]: 一批标准问答对列表
    """
    conn = mysql.connector.connect(**DB_CONFIG)
    try:
        with conn.cursor(dictionary=True) as cursor:
            # 创建用于查询的占位符 (%s) 
            placeholders = ', '.join(['%s'] * len(qa_ids))
            query = f"""
                SELECT r.id                 AS id,
                    q.source_question_id AS source_question_id,
                    a.source_answer_id   AS source_answer_id,
                    q.question           as question,
                    a.answer             as answer,
                    a.key_points         as key_points,
                    q.status_id          as q_status_id,
                    q.status_reason      as q_status_reason,
                    a.status_id          as a_status_id,
                    a.status_reason      as a_status_reason
                FROM standard_qa_relations r
                        JOIN standard_questions q on r.question_id = q.id
                        JOIN standard_answers a on r.answer_id = a.id
                WHERE r.id in ({placeholders})
                AND ((SELECT COUNT(*)
                        FROM model_eval_answers temp_mea
                        WHERE temp_mea.qa_id = r.id
                        AND temp_mea.model_id = %s) <= 0)
                AND ((q.status_id = 1 or q.status_id = 2 or q.status_id = 4) AND
                    (a.status_id = 1 or a.status_id = 2 or a.status_id = 4))
                LIMIT %s, %s
            """
            # 执行查询时将参数传递给 execute
            cursor.execute(query, tuple(qa_ids) + (llm_model_id, offset, limit))
            return [StandardQA.from_db_row(row) for row in cursor.fetchall()]
    finally:
        conn.close()

def get_saved_model_answer(qa_id: int, llm_model_id: str) -> ModelAnswer:
    """
    从数据库获取已保存的被测大模型回答
    
    Args:
        qa_id (int): 问答对的ID
        llm_model_id (str): 大语言模型的ID
        
    Returns:
        ModelAnswer: 模型回答对象，如果不存在则返回None
    """
    conn = mysql.connector.connect(**DB_CONFIG)
    try:
        logger.info(f"正在获取已保存的模型回答: qa_id={qa_id}, llm_model_id={llm_model_id}")    
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

def fetch_model_answer(modelConfig: ModelConfig, llm: LLM, qa: StandardQA) -> ModelAnswer:
    """
    通过大模型Api获取被测大模型对问题的回答

    Args:
        modelConfig (ModelConfig): 被测大模型的配置信息
        llm (LLM): 被测大模型的客户端实例
        qa (StandardQA): 待回答的标准问答对

    Returns:
        ModelAnswer: 包含模型回答内容的对象
    """
    reasoning_content, answer_content =  llm.autoCall(
        system_role=EVALUATED_SYSTEM_PROMPT,
        user_content=qa.split_question,
        temperature=0.2,
        max_tokens=4096,
        log_output=False,
    )
    modelAnswer = ModelAnswer.from_model_answer(
        qa_id=qa.id,
        llm_model_id=modelConfig.id,
        model_name=modelConfig.model_name, 
        model_id=modelConfig.model_id,
        model_remark=modelConfig.model_remark,
        answer_content=answer_content, 
        reasoning_content=reasoning_content
    )
    model_answer_info = modelAnswer.format_simple_info()
    logger.info(f"\nQ: {qa.split_question}\nA: {model_answer_info}")
    return modelAnswer

def save_model_answers_batch(model_config: ModelConfig, qa_answers: List[Tuple[StandardQA, ModelAnswer]], api_key_encryptor: ApiKeyEncryptor) -> List[ModelAnswer]:
    """
    批量保存被测大模型的回答到数据库
    
    Args:
        model_config (ModelConfig): 被测大模型的配置
        qa_answers (List[Tuple[StandardQA, ModelAnswer]]): 问答对和对应的模型回答列表
        api_key_encryptor (ApiKeyEncryptor): API密钥加密器
        
    Returns:
        List[ModelAnswer]: 保存后的模型回答列表
    """
    if not qa_answers:
        return []
    
    # 过滤出需要保存的新回答
    new_answers = [(qa, model_answer) for qa, model_answer in qa_answers if model_answer.is_new]
    if not new_answers:
        return [model_answer for _, model_answer in qa_answers]
    
    conn = mysql.connector.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cursor:
            # 确保模型存在
            llm_model_id = model_config.id
            cursor.execute("""
                SELECT id FROM llm_models WHERE id = %s
            """, (llm_model_id,))
            result = cursor.fetchone()
            if not result or llm_model_id == -1:
                cursor.execute("""
                    INSERT INTO llm_models 
                    (model_name, model_id, model_remark, base_url, encrypted_api_key, api_type, model_type)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (model_config.to_encrypted_config(api_key_encryptor)))
                llm_model_id = cursor.lastrowid
                model_config.id = llm_model_id
            
            # 批量插入新的模型回答
            values = [
                (model_answer.qa_id, llm_model_id, model_answer.reasoning_content, model_answer.answer_content)
                for _, model_answer in new_answers
            ]
            
            cursor.executemany("""
                INSERT INTO model_eval_answers
                (qa_id, model_id, reasoning_content, answer_content)
                VALUES (%s, %s, %s, %s)
            """, values)
            
            # 获取插入的ID并更新新插入的ModelAnswer对象
            last_id = cursor.lastrowid
            for i, (_, model_answer) in enumerate(new_answers):
                model_answer.id = last_id - len(new_answers) + i + 1
            
            conn.commit()
            return [model_answer for _, model_answer in qa_answers]
    finally:
        conn.close()

def fetch_model_answers_concurrently(model_config: ModelConfig, qa_list: List[StandardQA], max_workers: int = 3) -> List[Tuple[StandardQA, ModelAnswer]]:
    """
    并发获取一批问答对的模型回答
    
    Args:
        model_config (ModelConfig): 被测模型的配置
        qa_list (List[StandardQA]): 问答对列表
        max_workers (int): 最大并发工作线程数(默认3)
        
    Returns:
        List[Tuple[StandardQA, ModelAnswer]]: 问答对和对应的模型回答列表
    """
    results = []
    llm = LLM(model_config)
    
    def process_qa(qa: StandardQA) -> Tuple[StandardQA, ModelAnswer]:
        try:
            # 检查是否已有保存的回答
            model_answer = get_saved_model_answer(qa.id, model_config.id)
            if model_answer:
                # 标记为已存在的回答, 不需要保存
                model_answer.is_new = False
            else:
                # 获取模型对该问题的回答，标记为新回答，需要保存
                model_answer = fetch_model_answer(model_config, llm, qa)
                model_answer.is_new = True
            return qa, model_answer
        except Exception as e:
            logger.error(f"处理问答对 {qa.id} 时出错, 模型: {model_config}, 错误: {e}", exc_info=True)
            return qa, None
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_qa = {executor.submit(process_qa, qa): qa for qa in qa_list}
        for future in as_completed(future_to_qa):
            qa, model_answer = future.result()
            if model_answer:
                results.append((qa, model_answer))
    
    return results

@dataclass
class FetchModelParams:
    """获取模型回答参数类"""
    # 模型配置
    model_config: ModelConfig
    # 限制获取问答对数量
    limit_qa_count: int
    # 每次获取问答对数量
    batch_size: int
    # 问答对工作线程数
    qa_workers: int
    # 问答对来源
    qa_src: str
    # 问答对ID列表
    qa_id_list: List[int]
    # API密钥加解密器
    api_key_encryptor: ApiKeyEncryptor

@timer
def fetch_model_answers(params: FetchModelParams):
    """
    获取模型回答
    
    Args:
        params (FetchModelParams): 获取模型回答参数
    """
    # 准备被测模型信息
    model_logger_str = params.model_config.format_single_info()
    # 使用ErrorCounter来记录错误次数, 
    # 如果一个模型在获取回答时出错次数超过MAX_API_ERROR_LIMIT次, 则停止对该模型的评估
    errorCounter: ErrorCounter = ErrorCounter(max_errors=MAX_API_ERROR_LIMIT)    

    def get_next_batch()->Tuple[bool, List[StandardQA]]:
        """
        根据配置的数据源类型（列表或批量），获取指定批次大小的未回答问答对数据。
        成功则返回True和问答对列表，失败则返回False和None。
        Args:
            offset: 偏移量，用于分页查询
        Returns:
            Tuple[bool, List[StandardQA]]: 成功标志和问答对列表
        """
        # 偏移量，设定为0，因为每次保存后，未回答的问答对会动态变化
        offset = 0
        try:
            if params.qa_src == 'list':
                return True, get_unanswered_standard_qa(
                    qa_ids=params.qa_id_list, offset=offset, 
                    limit=params.batch_size, 
                    llm_model_id=params.model_config.id
                )
            else:
                return True, get_unanswered_standard_qa_batch(
                    offset=offset, limit=params.batch_size, 
                    llm_model_id=params.model_config.id
                )
        except Exception as e:
            logger.error(f"获取问答对时出错: {e} 被测模型: {model_logger_str}", exc_info=True)
            return False, []
    
    logger.info(f"开始获取模型回答: {model_logger_str}")
    
    # 批量获取batch_size个问答对，然后让大模型回答这些问题
    processed_count = 0  # 已处理的问答对数量
    while processed_count < params.limit_qa_count:
        # 获取下一批未回答的问答对
        success, qa_list = get_next_batch()
        if not success:
            logger.error(f"获取问答对失败!被测模型: {model_logger_str}")
            break
        if not qa_list:
            logger.info(f"没有更多问答对可获取. 被测模型: {model_logger_str}")
            break
        # 处理查询到的问答对数量，以保证不超过限制数量
        remaining_count = params.limit_qa_count - processed_count
        if len(qa_list) > remaining_count:
            logger.info(f"获取到的问答对数量({len(qa_list)})超过剩余限制({remaining_count}), 仅使用前{remaining_count}个问答对")
            qa_list = qa_list[:remaining_count]
        
        # 更新已处理的问答对数量
        processed_count += len(qa_list)
        # 获取回答并保存
        qa_logger_str = f"问答对 (共{len(qa_list)}个) : {[qa.id for qa in qa_list]}"
        logger.info(f"开始获取回答并保存... 被测模型: {model_logger_str}, {qa_logger_str}")
        try: 
            # 获取模型回答
            qa_answers = fetch_model_answers_concurrently(params.model_config, qa_list, max_workers=params.qa_workers)
            # 保存模型回答
            qa_answers = save_model_answers_batch(params.model_config, qa_answers, params.api_key_encryptor)
            logger.info(f"回答保存完成. 被测模型: {model_logger_str}, {qa_logger_str}")
        except Exception as e:
            logger.error(f"处理批次时出错: {e} 被测模型: {model_logger_str}, {qa_logger_str}", exc_info=True)
            if errorCounter.count_and_check_max_error():
                logger.warning(f"错误次数超过阈值, 停止获取回答. 被测模型: {model_logger_str}, {qa_logger_str}")
                break
            else:
                continue

def parse_args():
    """
    解析命令行参数
    
    Returns:
        argparse.Namespace: 解析后的参数对象
    """
    parser = argparse.ArgumentParser(
        description='获取模型回答工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  1. 从所有问答对中获取回答:
     python llm_qa_eval_fetch.py --qa-src all
  
  2. 从指定问答对中获取回答:
     python llm_qa_eval_fetch.py --qa-src list --qa-id-list 1
     python llm_qa_eval_fetch.py --qa-src list --qa-id-list 1,2,3,4,5
  
  3. 调整并发和批处理参数: 同时并发测试5个大模型，限制每个大模型只回答88个问题，每批从数据库中获取的问题数量为44，每个大模型同时并发回答的问题数量为4
     python llm_qa_eval_fetch.py --model-workers 5 --limit-qa-count 88 --batch-size 44 --qa-workers 4
  
  4. 启用debug模式:
     python llm_qa_eval_fetch.py --debug --qa-src list --qa-id-list 5
  
  5. 组合使用:
     python llm_qa_eval_fetch.py --model-workers 5 --limit-qa-count 50 --batch-size 4 --qa-workers 1 --qa-src all
     python llm_qa_eval_fetch.py --model-workers 1 --limit-qa-count 1 --batch-size 1 --qa-workers 1 --qa-src list --qa-id-list 5
"""
    )
    
    # 默认参数
    DEFAULT_MODEL_WORKERS = 1
    DEFAULT_LIMIT_QA_COUNT = 8
    DEFAULT_BATCH_SIZE = 8
    DEFAULT_QA_WORKERS = 2
    DEFAULT_QA_SRC = 'all'

    # 被测大模型并发数
    parser.add_argument(
        '--model-workers',
        type=int,
        default=DEFAULT_MODEL_WORKERS,
        help=f'同时并发运行的大模型数量 (默认: {DEFAULT_MODEL_WORKERS})'
    )

    # 限制每个大模型回答的问题总数
    parser.add_argument(
        '--limit-qa-count',
        type=int,
        default=DEFAULT_LIMIT_QA_COUNT,
        help=f'限制每个大模型回答的问题总数 (默认: {DEFAULT_LIMIT_QA_COUNT})'
    )
    
    # 每批从数据库中获取的问题数量
    parser.add_argument(
        '--batch-size',
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f'每批从数据库中获取的问题数量 (默认: {DEFAULT_BATCH_SIZE})'
    )
    
    # 每个被测大模型并发回答的问题数量
    parser.add_argument(
        '--qa-workers',
        type=int,
        default=DEFAULT_QA_WORKERS,
        help=f'每个被测大模型同时并发回答的问题数量 (默认: {DEFAULT_QA_WORKERS})'
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

    # 调试模式
    parser.add_argument(
        '--debug',
        action='store_true',
        help='启用调试模式（默认关闭），包括使用秒级时间戳记录日志'
    )

    # 解析命令行参数    
    return parser.parse_args()

@timer  
def main():
    """
    主函数，负责解析命令行参数并执行模型回答获取任务
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
        
        # 处理问答对来源参数
        qa_id_list = []
        if args.qa_src == 'list':
            if not args.qa_id_list:
                raise ValueError("使用 --qa-src list 时必须指定 --qa-id-list")
            qa_id_list = [int(id) for id in args.qa_id_list.split(',')]
        
        # 输出运行参数
        logger.info("运行参数配置: ")
        logger.info(f"  同时并发运行的大模型数量(model-workers): {args.model_workers}")
        logger.info(f"  限制每个大模型回答的问题总数(limit-qa-count): {args.limit_qa_count}")
        logger.info(f"  每批从数据库中获取的问题数量(batch-size): {args.batch_size}")
        logger.info(f"  每个被测大模型同时并发回答的问题数量(qa-workers): {args.qa_workers}")
        logger.info(f"  问答对来源(qa-src): {args.qa_src }")
        if args.qa_src  == 'list':
            logger.info(f"  问答对ID列表(qa-id-list): {qa_id_list}")
        
        # 被测模型配置文件路径
        EVAL_MODELS_CONFIG_FILE_PATH = CONFIG[CONFIG_NAME]['eval_models_config_file_path']
        ABS_EVAL_MODELS_CONFIG_FILE_PATH = os.path.abspath(EVAL_MODELS_CONFIG_FILE_PATH)
        
        # 初始化API密钥加密器
        logger.info(f"初始化API加解密器")
        apiKeyEncryptor = ApiKeyEncryptor(get_api_key_secret_key(CONFIG))
        
        # 加载被测模型配置
        logger.info(f"正在从 {ABS_EVAL_MODELS_CONFIG_FILE_PATH} 加载被测模型配置")
        evalModelConfigList = load_models_configs(
            apiKeyEncryptor=apiKeyEncryptor,
            db_config=DB_CONFIG,
            file_path=ABS_EVAL_MODELS_CONFIG_FILE_PATH
        )
        logger.info(f"已加载 {len(evalModelConfigList)} 个被测模型配置: {[sModelConfig.format_single_info() for sModelConfig in evalModelConfigList]}")
        
        # 使用线程池并发评估多个模型
        # 优化max_workers设置：取用户指定的worker数量和实际模型数量的最小值
        max_workers = min(args.model_workers, len(evalModelConfigList))
        logger.info(f"线程池配置: 用户指定worker数={args.model_workers}, 实际模型数={len(evalModelConfigList)}, 最终max_workers={max_workers}")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for sModelConfig in evalModelConfigList:
                futures.append(
                    executor.submit(
                        fetch_model_answers,
                        FetchModelParams(
                            model_config=sModelConfig,
                            qa_workers=args.qa_workers,
                            batch_size=args.batch_size,
                            limit_qa_count=args.limit_qa_count,
                            qa_src=args.qa_src,
                            qa_id_list=qa_id_list,
                            api_key_encryptor=apiKeyEncryptor
                        )
                    )
                )
            
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"获取模型回答过程中出错: {e}", exc_info=True)
                    
    except Exception as e:
        logger.error(f"执行出错: {e}", exc_info=True)

if __name__ == "__main__":
    main()