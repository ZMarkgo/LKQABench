import time
from functools import wraps
import logging

logger: logging.Logger = logging.getLogger(__name__)

def format_time_cost(seconds):
    """
    将秒数格式化为易读的时间字符串
    
    Args:
        seconds (float): 时间长度（秒）
        
    Returns:
        str: 格式化后的时间字符串，如"1分30.50秒"、"2小时15分30.25秒"等
        
    Examples:
        >>> format_time_cost(30.5)
        '30.50秒'
        >>> format_time_cost(90.25)
        '1分30.25秒'
        >>> format_time_cost(3665.75)
        '1小时1分5.75秒'
    """
    if seconds < 60:
        return f"{seconds:.2f}秒"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{minutes:.0f}分{seconds:.2f}秒"
    elif seconds < 86400:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f"{hours:.0f}小时{minutes:.0f}分{seconds:.2f}秒"
    else:
        days = seconds // 86400
        hours = (seconds % 86400) // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f"{days:.0f}天{hours:.0f}小时{minutes:.0f}分{seconds:.2f}秒"

def timer(func):
    """
    装饰器：测量函数执行时间并记录日志
    
    Args:
        func: 被装饰的函数
        
    Returns:
        function: 包装后的函数，会自动记录执行时间
        
    Examples:
        @timer
        def my_function():
            time.sleep(1)
            return "done"
            
        # 调用时会自动记录: "my_function 耗时: 1.00秒"
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 记录开始时间
        start = time.perf_counter()
        # 执行原函数
        result = func(*args, **kwargs)
        # 计算耗时
        time_cost_in_seconds = time.perf_counter() - start
        # 格式化并记录日志
        msg=f"{func.__name__} 耗时: {format_time_cost(time_cost_in_seconds)}"
        logger.info(msg)
        return result
    return wrapper

# 使用示例: 
if __name__ == "__main__":
    @timer
    def my_function():
        # 待测代码
        time.sleep(1)
        pass
    my_function()
