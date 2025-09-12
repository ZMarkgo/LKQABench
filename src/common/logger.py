import logging
from logging import Logger
from enum import IntEnum
from datetime import datetime
import os

class Level(IntEnum):
    """日志级别枚举类"""
    CRITICAL = logging.CRITICAL
    ERROR = logging.ERROR
    WARNING = logging.WARNING
    INFO = logging.INFO
    DEBUG = logging.DEBUG
    NOTSET = logging.NOTSET

class TimestampType(IntEnum):
    """时间戳类型枚举类"""
    SECOND = 1  # 精确到秒，便于调试
    DAY = 2     # 精确到天，便于分析

DEFAULT_LOG_FORMAT = "%(asctime)s - %(levelname)s - (%(funcName)s) %(message)s"
DEFAULT_LOG_LEVEL = Level.INFO
DEFAULT_FILE_MODE = 'a'
DEFAULT_ENCODING = 'utf-8'

def get_log_file_with_timestamp(log_file: str, timestamp_type: TimestampType = TimestampType.SECOND) -> str:
    """
    为日志文件添加时间戳后缀
    
    Args:
        log_file (str): 原始日志文件名
        timestamp_type (TimestampType): 时间戳类型，SECOND为精确到秒，DAY为精确到天
        
    Returns:
        str: 添加时间戳后的日志文件名
    """
    if timestamp_type == TimestampType.SECOND:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    elif timestamp_type == TimestampType.DAY:
        timestamp = datetime.now().strftime("%Y%m%d")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    file_name, file_ext = os.path.splitext(log_file)
    return f"{file_name}_{timestamp}{file_ext}"

def setup_logger(log_file: str, level: Level = DEFAULT_LOG_LEVEL, format: str = DEFAULT_LOG_FORMAT,
                 filemode: str = DEFAULT_FILE_MODE, encoding: str = DEFAULT_ENCODING, name: str = __name__, 
                 add_timestamp: bool = True, timestamp_type: TimestampType = TimestampType.DAY,
                 print_log_location: bool = True) -> Logger:
    """
    配置并返回一个日志记录器
    
    Args:
        log_file (str): 日志文件路径
        level (Level): 日志级别
        format (str): 日志格式
        filemode (str): 文件打开模式，'w'表示覆盖，'a'表示追加
        encoding (str): 文件编码
        name (str): 日志器名称
        add_timestamp (bool): 是否在日志文件名中添加时间戳后缀，默认为True
        timestamp_type (TimestampType): 时间戳类型，SECOND为精确到秒（调试用），DAY为精确到天（分析用），默认为DAY
        print_log_location (bool): 是否在控制台输出日志文件位置，默认为True
        
    Returns:
        logging.Logger: 配置好的日志记录器实例
    """
    # 如果启用时间戳，则在文件扩展名前添加时间戳
    if add_timestamp:
        log_file = get_log_file_with_timestamp(log_file, timestamp_type)
    
    # 如果需要，在控制台输出日志文件位置
    if print_log_location:
        abs_log_file = os.path.abspath(log_file)
        print(f"日志文件: {abs_log_file}")
    
    # 配置日志器，输出到文件
    logging.basicConfig(
        level=level,
        format=format,
        filename=log_file,
        filemode=filemode,
        encoding=encoding
    )
    return get_logger(name)

def get_logger(name: str = __name__) -> Logger:
    return logging.getLogger(name)

# # 用法一：调试场景 - 使用秒级时间戳
# debug_logger = setup_logger(
#     log_file='logs/debug_test.log',
#     level=Level.DEBUG,
#     add_timestamp=True,
#     timestamp_type=TimestampType.SECOND,
#     print_log_location=True,
#     name='debug_logger'
# )


# # 用法二：分析场景 - 使用天级时间戳
# analysis_logger = setup_logger(
#     log_file='logs/analysis_test.log',
#     level=Level.INFO,
#     add_timestamp=True,
#     timestamp_type=TimestampType.DAY,
#     print_log_location=True,
#     name='analysis_logger'
# )
