import logging
import sys
import os
from datetime import datetime

class Logger:
    def __init__(self, name="CodePretrain"):
        # 创建日志目录
        log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # 配置日志文件路径
        log_file = os.path.join(log_dir, f'code_pretrain_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        
        # 配置日志格式
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(log_file)
            ]
        )
        self.logger = logging.getLogger(name)

    def __call__(self, message):
        """允许直接调用实例来记录日志"""
        self.info(message)

    def info(self, message):
        self.logger.info(message)

    def error(self, message):
        self.logger.error(message)

    def warning(self, message):
        self.logger.warning(message)

    def debug(self, message):
        self.logger.debug(message)

# 创建全局logger实例
logger = Logger()

# 为了兼容之前的代码，我们也可以直接使用函数形式
def Logger(message):
    """兼容函数式调用"""
    logger(message)