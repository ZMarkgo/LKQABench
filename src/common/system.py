import logging
import sys

logger: logging.Logger = logging.getLogger(__name__)

def exit(code: int = 0):
    """Exit the interpreter by raising SystemExit(code).
    
    If code is omitted or None, it defaults to zero (i.e., success).
    If code is an integer, it specifies the system exit status.
    If code is another kind of object, it is printed and the system
    exit status is one (i.e., failure).
    """
    if code != 0:
        logger.error(f"程序异常退出，状态码：{code}...")
    else:
        logger.info(f"程序退出，状态码：{code}...")
    sys.exit(code)  # 修复参数传递方式，sys.exit()不接受关键字参数

class ErrorCounter:
    def __init__(self, max_errors=10):
        self.error_count = 0
        self.MAX_ERRORS = max_errors

    def count_and_check(self):
        """
        记录错误次数，检查错误次数是否超过阈值，如果超过则log并退出程序
        """
        self.error_count += 1
        if self.error_count > self.MAX_ERRORS:
            logger.warn(f"错误次数超过阈值：{self.MAX_ERRORS}，程序即将退出...")
            exit(code=1)

    def count_and_check_max_error(self):
        """
        记录错误次数，检查错误次数是否超过阈值，如果超过则返回True，否则返回False
        """
        self.error_count += 1
        return self.error_count > self.MAX_ERRORS
