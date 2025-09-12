from sentence_transformers import SentenceTransformer
from typing import Optional
import logging
import threading
from common.logger import get_logger

# 配置基础的控制台日志输出
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = get_logger
else:
    logger = get_logger(__name__)

class SentenceModelManager:
    """句向量模型管理器，使用简单的单例模式确保只加载一次模型"""
    _model: Optional[SentenceTransformer] = None
    _model_name: str = 'all-MiniLM-L6-v2'
    _lock = threading.Lock()  # 类级别的锁

    @classmethod
    def _load_model(cls):
        """加载模型的内部方法"""
        logger.info(f"开始加载句向量模型: {cls._model_name}")
        try:
            # 首先尝试从本地缓存加载
            cls._model = SentenceTransformer(cls._model_name, local_files_only=True)
            logger.info("从本地缓存加载句向量模型成功")
        except Exception as e:
            logger.info(f"本地缓存加载失败: {e}，尝试从网络下载...")
            # 如果本地没有，则从网络下载
            cls._model = SentenceTransformer(cls._model_name)
            logger.info("从网络下载句向量模型成功")
        logger.info("模型加载完成")



    @classmethod
    def get_model(cls) -> SentenceTransformer:
        """获取句向量模型实例"""
        if cls._model is None:
            with cls._lock:
                if cls._model is None:  # 双重检查
                    cls._load_model()
        return cls._model

    @classmethod
    def set_model_name(cls, model_name: str):
        """设置模型名称并重新加载模型"""
        with cls._lock:
            if model_name != cls._model_name:
                logger.info(f"切换模型: {cls._model_name} -> {model_name}")
                cls._model_name = model_name
                cls._model = None  # 重置模型，下次调用时会重新加载