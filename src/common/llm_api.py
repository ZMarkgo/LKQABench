from openai import OpenAI
import anthropic
import logging
from enum import Enum
import mysql.connector
import json
from configparser import ConfigParser
from typing import Union
from common import system
from common.api_key_encryptor import ApiKeyEncryptor

logger: logging.Logger = logging.getLogger(__name__)

REASONING_LOGGER_HEADER = "="*25 + "思考过程(Reasoning)" + "="*25
CONTENT_LOGGER_HEADER = "="*25 + "回答内容(Content)" + "="*25

class ModelType(Enum):
    """模型类型"""
    CHAT = 'chat'
    REASONING = 'reasoning'

class APIType(Enum):
    """API类型"""
    OPENAI = 'OpenAI'
    ANTHROPIC = 'Anthropic'
    QWEN = 'Qwen'
    DEEPSEEK = 'Deepseek'
    MOCK = 'Mock'

class ModelConfig:
    """模型的配置"""
    def __init__(self, id, model_name, model_id, api_key, base_url, model_remark, api_type, model_type, new_model=None):
        """
        :param id: 模型在数据库中的ID
        :param model_name: 模型代号
        :param model_id: 模型型号，用于发送请求
        :param api_key: 模型API Key
        :param base_url: 模型API Base URL
        :param model_remark: 模型备注，可以为空
        :param api_type: 模型API类型, 可选：'OpenAI', 'Anthropic', 'Qwen', 'Deepseek', 'Mock'
        :param model_type: 模型类型, 可选：'chat', 'reasoning'
        :param new_model: 是否为新模型，None时根据id自动判断（id<0为新模型）
        """
        self.id = id
        self.model_name = model_name
        self.model_id = model_id
        self.api_key = api_key
        self.base_url = base_url
        self.model_remark = model_remark
        self.apiType:APIType = APIType(api_type)
        self.modelType:ModelType = ModelType(model_type)
        
        # 设置新模型标记
        if new_model is None:
            self._is_new_model = id < 0
        else:
            self._is_new_model = new_model
    
    def is_new_model(self) -> bool:
        """
        判断模型是否为新模型
        :return: 如果模型ID小于0，则返回True；否则返回False
        """
        return self._is_new_model

    @classmethod
    def build(cls, config: Union[ConfigParser, dict], encryptor: ApiKeyEncryptor = None, db_config: dict = None):
        """
        根据配置构建ModelConfig实例
        
        该方法支持两种模式：
        1. 直接从配置文件创建新的模型配置（当new_model=True时）
        2. 从数据库读取已有的模型配置（当提供llm_model_id时，需要encryptor解密API Key）
        
        :param config: 配置对象，包含模型配置信息或数据库模型ID {model_name, model_id, api_key, base_url, model_remark, api_type, model_type}
        :type config: Union[ConfigParser, dict]
        :param encryptor: API Key加密器，用于从数据库读取模型配置时解密API Key
        :type encryptor: ApiKeyEncryptor
        :param db_config: 数据库配置，用于从数据库读取模型配置
        :type db_config: dict
        :return: ModelConfig实例
        :raises ValueError: 当配置参数不足or需要encryptor和db_config但其为空时抛出异常
        """
        # 处理不同类型的配置
        if isinstance(config, ConfigParser):
            new_model = config.getboolean('new_model', False)
        else:  # dict类型
            new_model = config.get('new_model', False)
            if isinstance(new_model, str):
                new_model = new_model.lower() == 'true'

        if new_model:
            # 使用配置文件中的参数直接创建新的模型配置
            config_dict = dict(config)
            config_dict['new_model'] = True
            return cls.from_full_config(config_dict)
        elif 'llm_model_id' in config:
            # 使用数据库中已有的模型配置
            if encryptor is None:
                raise ValueError("encryptor 不能为空")
            if db_config is None:
                raise ValueError("db_config 不能为空")
            return cls.from_db(db_config, config['llm_model_id'], encryptor)
        else:
            # 配置不完整，无法创建模型配置
            if isinstance(config, ConfigParser):
                config_dict = {section: dict(config[section]) for section in config.sections()}
                error_msg = f"配置文件中缺少必要的参数 llm_model_id，当前配置内容：{config_dict}"
            else:
                error_msg = f"配置文件中缺少必要的参数 llm_model_id，当前配置内容：{config}"
            raise ValueError(error_msg)

    @classmethod
    def from_full_config(cls, config:dict):
        """
        从完整的配置中读取模型配置
        :param config: 模型配置 {model_name, model_id, api_key, base_url, model_remark, api_type, model_type, new_model}
        其中：
            model_name: 模型代号
            model_id: 模型型号，用于发送请求
            api_key: 模型API Key, 明文
            base_url: 模型API Base URL
            model_remark: 模型备注, 可以为空
            api_type: 模型API类型, 可选：'OpenAI', 'Anthropic', 'Qwen', 'Deepseek'
            model_type: 模型类型, 可选：'chat','reasoning'
            new_model: 是否为新模型，None时根据id自动判断（id<0为新模型）
        """
        return cls(
            id=config.get('id', -1),
            model_name=config['model_name'],
            model_id=config['model_id'],
            api_key=config['api_key'],
            base_url=config['base_url'],
            model_remark=config.get('model_remark', ''),
            api_type=config['api_type'],
            model_type=config['model_type'],
            new_model=config.get('new_model')
        )

    @classmethod
    def from_encrypted_config(cls, config:dict, encryptor: ApiKeyEncryptor):
        """
        从加密的配置中读取模型配置
        :param config: 加密的模型配置 {model_name, model_id, encrypted_api_key, base_url, model_remark, api_type, model_type}
        :param encryptor: ApiKeyEncryptor实例，用于解密API Key
        :return: ModelConfig实例
        """
        decrypted_config = dict(config)
        decrypted_config['api_key'] = encryptor.decrypt(config['encrypted_api_key'])
        # 从加密配置创建的实例标记为非新模型（通常来自数据库）
        decrypted_config['new_model'] = False
        return cls.from_full_config(decrypted_config)

    @classmethod
    def from_config(cls, config:ConfigParser):
        """
        准备废弃，改用 from_full_config
        """
        return cls.from_full_config(config)

    @classmethod
    def from_db(cls, db_config:dict, llm_model_id:int, encryptor:ApiKeyEncryptor):
        """
        根据id从数据库中读取模型配置
        :param db_config: 数据库配置
        :param llm_model_id: 数据库中的模型ID，不是model_id字段
        :param encryptor: ApiKeyEncryptor实例，用于解密API Key
        :return: ModelConfig实例
        """
        conn = mysql.connector.connect(**db_config)
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT model_name,
                        model_id,
                        model_remark,
                        base_url,
                        encrypted_api_key,
                        api_type,
                        model_type
                    FROM llm_models
                    WHERE id = %s
                """, (llm_model_id,))
                result = cursor.fetchone()
                if result is None:
                    raise ValueError(f"模型（id={llm_model_id}） 不存在")
                # 将查询结果转换为字典格式
                config_dict = {
                    'id': llm_model_id,
                    'model_name': result[0],
                    'model_id': result[1],
                    'model_remark': result[2],
                    'base_url': result[3],
                    'encrypted_api_key': result[4],
                    'api_type': result[5],
                    'model_type': result[6]
                }
                return cls.from_encrypted_config(config_dict, encryptor)
        finally:
            conn.close()  # 确保关闭连接

    def get_encrypted_api_key(self, encryptor: ApiKeyEncryptor) -> str:
        """
        获取加密的API Key
        :param encryptor: ApiKeyEncryptor实例，用于加密API Key
        :return: 加密的API Key
        """
        return encryptor.encrypt(self.api_key)

    def to_encrypted_config(self, encryptor: ApiKeyEncryptor) -> dict:
        """
        生成可存储的加密配置
        :param encryptor: ApiKeyEncryptor实例，用于加密API Key
        :return: 包含加密API Key的配置字典 {
            model_name: 模型代号
            model_id: 模型型号，用于发送请求
            model_remark: 模型备注，可以为空
            encrypted_api_key: 加密的API Key
            base_url: 模型API Base URL
            api_type: 模型API类型, 可选：'OpenAI', 'Anthropic', 'Qwen', 'Deepseek'
            model_type: 模型类型, 可选：'chat','reasoning'
        """
        return {
            'model_name': self.model_name,
            'model_id': self.model_id,
            'model_remark': self.model_remark,
            'base_url': self.base_url,
            'encrypted_api_key': encryptor.encrypt(self.api_key),
            'api_type': self.apiType.value,
            'model_type': self.modelType.value
        }

    def __str__(self):
        return f"ModelConfig(id={self.id}, model_name={self.model_name}, model_id={self.model_id}, model_remark={self.model_remark})"

    def __repr__(self):
        return self.__str__()
    
    def format_single_info(self):
        return f"{self.model_name}({self.id})"
    
    def to_output_format(self) -> dict:
        """
        转换为输出格式的字典
        :return: 包含llm_model_id, model_name, model_remark的字典
        """
        return {
            "llm_model_id": self.id,
            "model_name": self.model_name,
            "model_remark": self.model_remark or ""
        }
    
    @staticmethod
    def get_new_models_info(new_model_configs: list['ModelConfig']) -> str:
        """
        获取新增模型的ID列表和详细配置的JSON格式字符串
        :param new_model_configs: 新增的模型配置列表
        :return: 格式化的字符串，包含ID列表和JSON配置
        """
        if not new_model_configs:
            return "没有新增的模型配置"
        
        # 生成新增模型的ID列表
        new_model_ids = [config.id for config in new_model_configs]
        id_list_str = f"新增模型ID列表: {new_model_ids}"
        
        # 生成详细配置的JSON格式
        models_output = {
            "models": [config.to_output_format() for config in new_model_configs]
        }
        
        json_str = json.dumps(models_output, ensure_ascii=False, indent=4)
        
        return f"{id_list_str}\n新增模型详细配置:\n{json_str}"

def load_models_configs(apiKeyEncryptor: ApiKeyEncryptor, db_config:dict, file_path:str) -> list[ModelConfig]:
    """从JSON文件加载多个模型配置"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            return [ModelConfig.build(
                        config=model, 
                        encryptor=apiKeyEncryptor, 
                        db_config=db_config
                    ) for model in config['models']]
    except FileNotFoundError:
        logger.error(f"错误: 找不到配置文件 {file_path}")
        system.exit(1)
    except json.JSONDecodeError:
        logger.error(f"错误: 配置文件 {file_path} 格式无效")
        system.exit(1)
    except KeyError:
        logger.error(f"错误: 配置文件 {file_path} 结构无效")
        system.exit(1)

class IChatLLM:
    """
    用于调用非推理大模型的接口
    """
    def __init__(self):
        pass
    
    def call(self, system_role: str, user_content: str, 
             temperature:float=0.3, max_tokens:int=4096, 
             response_in_json:bool=False, 
             log_output:bool=False) -> str:
        """
        调用非推理模型的抽象方法

        Args:
            system_role (str): 系统角色描述
            user_content (str): 用户内容
            temperature (float): 温度参数
            max_tokens (int): 最大token数
            response_in_json (bool): 是否返回JSON格式
            log_output (bool): 是否记录模型输出内容，默认不记录

        Returns:
            str: 返回最终答案
        """
        raise NotImplementedError("未实现的IChatLLM")

class IReasoningLLM:
    """
    用于调用推理大模型的接口
    """
    def __init__(self):
        pass

    def callReasoning(self, system_role: str, user_content: str, 
                      temperature:float=0.3, max_tokens:int=4096, 
                      response_in_json:bool=False, 
                      log_output:bool=False) -> tuple[str, str]:
        """
        调用推理模型的抽象方法，推理模型不支持 System Role等参数
        
        Args:
            system_role (str): 系统角色描述
            user_content (str): 用户内容
            temperature (float): 温度参数
            max_tokens (int): 最大token数
            response_in_json (bool): 是否返回JSON格式
            log_output (bool): 是否使用日志记录模型输出内容（content和reasoning），默认不记录
            
        Returns:
            tuple[str, str]: 返回(reasoning_content, content)元组
        """
        raise NotImplementedError("未实现的IReasoningLLM")

class ILLM:
    """
    用于调用LLM的接口
    """
    def __init__(self):
        pass

    def format_config_info(self):
        """
        格式化模型配置信息
        :return: 格式化后的模型配置信息
        """
        raise NotImplementedError("未实现的ILLM")

    def get_model_config(self) -> ModelConfig:
        """
        获取模型配置
        :return: 模型配置ModelConfig
        """
        raise NotImplementedError("未实现的ILLM")

    def autoCall(self, system_role: str, user_content: str, 
                 temperature:float=0.3, max_tokens:int=4096, 
                 response_in_json:bool=False, 
                 log_output:bool=False):
        """
        根据模型类型智能决定调用哪个API
        
        Args:
            system_role (str): 系统角色描述
            user_content (str): 用户内容
            temperature (float): 温度参数
            max_tokens (int): 最大token数
            response_in_json (bool): 是否返回JSON格式
            log_output (bool): 是否使用日志记录模型输出内容（content和reasoning），默认不记录
            
        Returns:
            tuple: CHAT 模型调用 call() 方法，返回 (None, content)；
                  REASONING 模型调用 callReasoning() 方法，返回 (reasoning_content, content)
        """
        raise NotImplementedError("未实现的ILLM")

class LLMBase(IChatLLM, IReasoningLLM, ILLM):
    """
    用于调用LLM的抽象类
    """
    def __init__(self, modelConfig:ModelConfig):
        IChatLLM.__init__(self)
        IReasoningLLM.__init__(self)
        ILLM.__init__(self)
        self.modelConfig = modelConfig
    
    def format_config_info(self):
        """
        格式化模型配置信息
        :return: 格式化后的模型配置信息
        """
        raise self.modelConfig.__str__()
    
    def get_model_config(self) -> ModelConfig:
        """
        获取模型配置
        :return: 模型配置ModelConfig
        """
        return self.modelConfig
    
    def autoCall(self, system_role: str, user_content: str, 
                 temperature:float=0.3, max_tokens:int=4096, 
                 response_in_json:bool=False, 
                 log_output:bool=False) -> tuple[str, str]:
        """
        根据模型类型决定调用哪个API
        
        Args:
            system_role (str): 系统角色描述
            user_content (str): 用户内容
            temperature (float): 温度参数
            max_tokens (int): 最大token数
            response_in_json (bool): 是否返回JSON格式
            log_output (bool): 是否使用日志记录模型输出内容（content和reasoning），默认不记录
            
        Returns:
            tuple: CHAT 模型调用 call() 方法，返回 (None, content)；
                  REASONING 模型调用 callReasoning() 方法，返回 (reasoning_content, content)
        """
        if self.modelConfig.modelType == ModelType.CHAT:
            reasoning_content, content = None, self.call(system_role=system_role, user_content=user_content, 
                                   temperature=temperature, 
                                   max_tokens=max_tokens,
                                   response_in_json=response_in_json,
                                   log_output=log_output)
        elif self.modelConfig.modelType == ModelType.REASONING:
            reasoning_content, content = self.callReasoning(system_role=system_role, user_content=user_content, 
                                   temperature=temperature, 
                                   max_tokens=max_tokens,
                                   response_in_json=response_in_json,
                                   log_output=log_output)
        else:
            raise ValueError(f"不支持的模型类型: {self.modelConfig.modelType}")
            
        # 根据log_output参数决定是否记录模型输出
        if log_output:
            if reasoning_content:
                logger.info(f"模型推理过程:\n{reasoning_content}")
            logger.info(f"模型输出内容:\n{content}")
            
        return reasoning_content, content

class DeepSeekReasoningLLM(LLMBase):
    """
    参考：官网https://api-docs.deepseek.com/zh-cn/guides/reasoning_model
    """

    def __init__(self, modelConfig:ModelConfig):
        super().__init__(modelConfig)

    def callReasoning(self, system_role: str, user_content: str, 
                      temperature:float=0.3, max_tokens:int=4096, 
                      response_in_json:bool=False, 
                      log_output:bool=False) -> tuple[str, str]:
        """
        调用DeepSeek推理模型，推理模型不支持 System Role、temperature 等参数
        
        Args:
            system_role (str): 系统角色描述
            user_content (str): 用户内容
            temperature (float): 温度参数（DeepSeek不支持）
            max_tokens (int): 最大token数（DeepSeek不支持）
            response_in_json (bool): 是否返回JSON格式（DeepSeek不支持）
            log_output (bool): 是否使用日志记录模型输出内容（content和reasoning），默认不记录

        Returns:
            tuple: (reasoning_content, content) 推理内容和最终答案
            
        Raises:
            ValueError: 当必要参数缺失时
            Exception: API调用失败时
        """
        
        if not system_role and not user_content:
            raise ValueError("问题不能为空")
        question=f"{system_role}\n{user_content}"
        client = OpenAI(api_key=self.modelConfig.api_key, base_url=self.modelConfig.base_url)
        
        logger.info("调用DeepSeek推理模型")
        response = client.chat.completions.create(
            model=self.modelConfig.model_id,
            messages=[{"role": "user", "content": question}]
        )
        if log_output:
            logger.info(f"DeepSeek推理模型调用成功, 响应内容: \n{response}")
        
        # 检查响应是否有效
        if not hasattr(response, 'choices') or not response.choices or len(response.choices) == 0:
            raise Exception("API响应无效, 没有返回选择项(choices)")
        if not hasattr(response.choices[0], 'message') or not response.choices[0].message:
            raise Exception("API响应无效, 没有返回消息(choices[0].message)")
        if not hasattr(response.choices[0].message, 'content') or not response.choices[0].message.content:
            raise Exception("API响应无效, 返回内容(content)为空")
        
        # 提取模型响应
        content = response.choices[0].message.content
        # 提取模型的思维链/思考过程/推理过程）
        reasoning_content = None
        if hasattr(response.choices[0].message, 'reasoning_content') and response.choices[0].message.reasoning_content:
            reasoning_content = response.choices[0].message.reasoning_content
        else:
            # 尝试从 content 中提取 reasoning_content
            # e.g. <think>reasoning_content</think>
            think_start = content.find("<think>")
            think_end = content.find("</think>")
            if think_start != -1 and think_end != -1 and think_end > think_start:
                reasoning_content = content[think_start+len("<think>"):think_end]
                content = content[think_end+len("</think>"):]
            else:
                raise Exception("API响应无效, 推理模型未返回思考过程(reasoning_content)")

        # 根据log_output参数决定是否记录模型输出
        if log_output:
            logger.info(f"{REASONING_LOGGER_HEADER}\n{reasoning_content}")
            logger.info(f"{CONTENT_LOGGER_HEADER}\n{content}")
            
        return reasoning_content, content

    def call(self, system_role: str, user_content: str, 
             temperature:float=0.3, max_tokens:int=4096, 
             response_in_json:bool=False, 
             log_output:bool=False) -> str:
        raise NotImplementedError("推理模型不支持 System Role")

class OpenAILLM(LLMBase):
    """OpenAI API类型的模型"""
    def __init__(self, modelConfig:ModelConfig):
        super().__init__(modelConfig)

    def callReasoning(self, system_role: str, user_content: str, 
                      temperature:float=0.3, max_tokens:int=4096, 
                      response_in_json:bool=False, 
                      log_output:bool=False) -> tuple[str, str]:
        client = OpenAI(api_key=self.modelConfig.api_key, base_url=self.modelConfig.base_url)

        try:
            if response_in_json:
                logger.info(f"调用OpenAI模式的推理模型 API (Responses API), max_tokens={max_tokens}, 启用json格式输出")
                response = client.responses.create(
                    model=self.modelConfig.model_id,
                    input=[
                        {"role": "system", "content": system_role},
                        {"role": "user", "content": user_content}
                    ],
                    # temperature=temperature,
                    max_output_tokens=max_tokens,
                    response_format={"type": "json_object"},
                )
            else:
                logger.info(f"调用OpenAI模式的推理模型 API (Responses API), max_tokens={max_tokens}, 不启用json格式输出")
                response = client.responses.create(
                    model=self.modelConfig.model_id,
                    input=[
                        {"role": "system", "content": system_role},
                        {"role": "user", "content": user_content}
                    ],
                    # temperature=temperature,
                    max_output_tokens=max_tokens,
                )
            # logger.info(response.output)

            reasoning_content = ""
            answer_content = ""

            for item in response.output:
                if item.type == "reasoning":
                    if hasattr(item, "content") and item.content:
                        reasoning_content += "".join([c.text for c in item.content if hasattr(c, "text")])
                elif item.type == "message" and item.role == "assistant":
                    if hasattr(item, "content") and item.content:
                        answer_content += "".join([c.text for c in item.content if hasattr(c, "text")])

            # 如果只需要最终回答，可以直接用 response.output_text
            if not answer_content:
                answer_content = response.output_text

            # NOTE: o4-mini 官方隐藏了思考过程，reasoning_content为空
            # 根据log_output参数决定是否记录模型输出
            if log_output:
                logger.info(f"{REASONING_LOGGER_HEADER}\n{reasoning_content}")
                logger.info(f"{CONTENT_LOGGER_HEADER}\n{answer_content}")

            return reasoning_content, answer_content
        
        except Exception as e:
            err_msg = str(e)
            # 检测是否为 Responses API 未实现的情况
            if "not implemented" in err_msg or "convert_request_failed" in err_msg:
                logger.warning("Responses API 不受支持，回退到 Chat Completions API")

                try:
                    question = f"{system_role}\n{user_content}"
                    response = client.chat.completions.create(
                        model=self.modelConfig.model_id,
                        messages=[{"role": "user", "content": question}]
                    )

                    logger.info(response.choices[0].message)

                    reasoning_content = getattr(response.choices[0].message, "reasoning_content", "")
                    content = response.choices[0].message.content

                    # 根据log_output参数决定是否记录模型输出
                    if log_output:
                        logger.info(f"{REASONING_LOGGER_HEADER}\n{reasoning_content}")
                        logger.info(f"{CONTENT_LOGGER_HEADER}\n{content}")

                    return reasoning_content, content

                except Exception as e2:
                    raise Exception(f"回退到 Chat Completions 调用失败: {str(e2)}")
            else:
                # 其他异常直接抛出
                raise Exception(f"调用OpenAI API失败: {str(e)}")

    def call(self, system_role: str, user_content: str, 
             temperature:float=0.3, max_tokens:int=4096, 
             response_in_json:bool=False, 
             log_output:bool=False) -> str:
        client = OpenAI(api_key=self.modelConfig.api_key, base_url=self.modelConfig.base_url)

        try:
            if response_in_json:
                logger.info(f"调用OpenAI模式的非推理模型 API, temperature={temperature}, max_tokens={max_tokens}, 启用json格式输出")
                response = client.chat.completions.create(
                    model=self.modelConfig.model_id,
                    messages=[
                        {"role": "system", "content": system_role},
                        {"role": "user", "content": user_content}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                )
            else:
                logger.info(f"调用OpenAI模式的非推理模型 API, temperature={temperature}, max_tokens={max_tokens}, 不启用json格式输出")
                response = client.chat.completions.create(
                    model=self.modelConfig.model_id,
                    messages=[
                        {"role": "system", "content": system_role},
                        {"role": "user", "content": user_content}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            answer = response.choices[0].message.content
            
            # 根据log_output参数决定是否记录模型输出
            if log_output:
                logger.info(f"{CONTENT_LOGGER_HEADER}\n{answer}")
                
            return answer

        except Exception as e:
            raise Exception(f"调用OpenAI API失败: {str(e)}")

class QwenLLM(LLMBase):
    """Qwen类型的模型，支持推理和非推理"""
    def __init__(self, modelConfig:ModelConfig):
        super().__init__(modelConfig)

    def callReasoning(self, system_role: str, user_content: str, 
                      temperature:float=0.3, max_tokens:int=4096, 
                      response_in_json:bool=False, 
                      log_output:bool=False) -> tuple[str, str]:
        client = OpenAI(api_key=self.modelConfig.api_key, base_url=self.modelConfig.base_url)

        try:
            if response_in_json:
                logger.info(f"调用Qwen模式的推理模型 API, temperature={temperature}, max_tokens={max_tokens}, 启用json格式输出")
                response = client.chat.completions.create(
                    model=self.modelConfig.model_id,
                    messages=[
                        {"role": "system", "content": system_role},
                        {"role": "user", "content": user_content}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                    extra_body={"enable_thinking": True},
                    stream=True,
                )
            else:
                logger.info(f"调用Qwen模式的推理模型 API, temperature={temperature}, max_tokens={max_tokens}, 不启用json格式输出")
                response = client.chat.completions.create(
                    model=self.modelConfig.model_id,
                    messages=[
                        {"role": "system", "content": system_role},
                        {"role": "user", "content": user_content}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_body={"enable_thinking": True},
                    stream=True,
                )

            reasoning_content = ""  # 完整思考过程
            answer_content = ""  # 完整回复
            is_answering = False  # 是否进入回复阶段
            for chunk in response:
                if not chunk.choices:
                    logger.info("\nUsage:\n" + chunk.usage)
                    continue

                delta = chunk.choices[0].delta

                # 只收集思考内容
                if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
                    reasoning_content += delta.reasoning_content

                # 收到content，开始进行回复
                if hasattr(delta, "content") and delta.content:
                    if not is_answering:
                        # logger.info("\n" + "=" * 20 + "思考过程" + "=" * 20 + "\n" + reasoning_content)
                        is_answering = True
                    answer_content += delta.content
            
            # 根据log_output参数决定是否记录模型输出
            if log_output:
                logger.info(f"{REASONING_LOGGER_HEADER}\n{reasoning_content}")
                logger.info(f"{CONTENT_LOGGER_HEADER}\n{answer_content}")
                
            return reasoning_content, answer_content

        except Exception as e:
            raise Exception(f"调用Qwen API失败: {str(e)}")

    def call(self, system_role: str, user_content: str, 
             temperature:float=0.3, max_tokens:int=4096, 
             response_in_json:bool=False, 
             log_output:bool=False) -> str:
        client = OpenAI(api_key=self.modelConfig.api_key, base_url=self.modelConfig.base_url)

        try:
            if response_in_json:
                logger.info(f"调用Qwen模式的非推理模型 API, temperature={temperature}, max_tokens={max_tokens}, 启用json格式输出")
                response = client.chat.completions.create(
                    model=self.modelConfig.model_id,
                    messages=[
                        {"role": "system", "content": system_role},
                        {"role": "user", "content": user_content}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                    extra_body={"enable_thinking": False},
                )
            else:
                logger.info(f"调用Qwen模式的非推理模型 API, temperature={temperature}, max_tokens={max_tokens}, 不启用json格式输出")
                response = client.chat.completions.create(
                    model=self.modelConfig.model_id,
                    messages=[
                        {"role": "system", "content": system_role},
                        {"role": "user", "content": user_content}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_body={"enable_thinking": False},
                )
            answer = response.choices[0].message.content
            
            # 根据log_output参数决定是否记录模型输出
            if log_output:
                logger.info(f"模型输出内容:\n{answer}")
                
            return answer

        except Exception as e:
            raise Exception(f"调用Qwen API失败: {str(e)}")

class AnthropicLLM(LLMBase):
    """Anthropic类型的非推理模型"""
    def __init__(self, modelConfig:ModelConfig):
        super().__init__(modelConfig)
    
    def callReasoning(self, system_role: str, user_content: str, 
                      temperature:float=0.3, max_tokens:int=4096, 
                      response_in_json:bool=False, 
                      log_output:bool=False) -> tuple[str, str]:
        raise NotImplementedError("非推理模型不支持 Reasoning")

    def call(self, system_role: str, user_content: str, 
             temperature:float=0.3, max_tokens:int=4096, 
             response_in_json:bool=False, 
             log_output:bool=False) -> str:
        logger.info("调用Anthropic模式的非推理模型 API")
        if response_in_json:
            logger.warn("Anthropic模式的非推理模型 API暂时没有实现 response_in_json")
        client = anthropic.Client(
            api_key=self.modelConfig.api_key,
            base_url=self.modelConfig.base_url
        )
        try:
            message = client.messages.create(
                model=self.modelConfig.model_id,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_role},
                    {"role": "user", "content": user_content}
                ],
                temperature=temperature,
            )
            answer = message.content[0].text
            
            # 根据log_output参数决定是否记录模型输出
            if log_output:
                logger.info(f"{CONTENT_LOGGER_HEADER}\n{answer}")
                
            return answer
        except Exception as e:
            raise Exception(f"调用Anthropic API失败: {str(e)}")

class MockLLM(LLMBase):
    """
    用于测试的模拟LLM，返回预设的回答
    """
    
    def __init__(self, modelConfig: ModelConfig, mock_content: str = "这是一个模拟回答", mock_reasoning: str = "这是模拟的推理过程"):
        """
        初始化MockLLM
        
        Args:
            modelConfig (ModelConfig): 模型配置
            mock_content (str): 模拟的回答内容
            mock_reasoning (str): 模拟的推理过程
        """
        super().__init__(modelConfig)
        self.mock_content = mock_content
        self.mock_reasoning = mock_reasoning

    def call(self, system_role: str, user_content: str, 
             temperature:float=0.3, max_tokens: int = 4096, 
             response_in_json: bool = False, 
             log_output:bool=False) -> str:
        """
        模拟调用非推理模型
        
        Args:
            system_role (str): 系统角色描述
            user_content (str): 用户内容
            temperature (float): 温度参数（模拟中忽略）
            max_tokens (int): 最大token数（模拟中忽略）
            response_in_json (bool): 是否返回JSON格式（模拟中忽略）
            log_output (bool): 是否记录模型输出内容，默认不记录
            
        Returns:
            str: 模拟的回答内容
        """
        answer = self.mock_content
        if response_in_json:
            answer = f'{{"answer": "{self.mock_content}"}}'
            
        # 根据log_output参数决定是否记录模型输出
        if log_output:
            logger.info(f"{CONTENT_LOGGER_HEADER}\n{answer}")
            
        return answer
        
    def callReasoning(self, system_role: str, user_content: str, 
                      temperature:float=0.3, max_tokens: int = 4096, 
                      response_in_json: bool = False, 
                      log_output:bool=False) -> tuple[str, str]:
        """
        模拟调用推理模型
        
        Args:
            system_role (str): 系统角色描述
            user_content (str): 用户内容
            temperature (float): 温度参数（模拟中忽略）
            max_tokens (int): 最大token数（模拟中忽略）
            response_in_json (bool): 是否返回JSON格式（模拟中忽略）
            log_output (bool): 是否使用日志记录模型输出内容（content和reasoning），默认不记录
            
        Returns:
            tuple[str, str]: (推理过程, 最终回答)
        """
        reasoning = self.mock_reasoning
        response = self.mock_content
        
        if response_in_json:
            response = f'{{"answer": "{response}"}}'
            
        # 根据log_output参数决定是否记录模型输出
        if log_output:
            logger.info(f"{REASONING_LOGGER_HEADER}\n{reasoning}")
            logger.info(f"{CONTENT_LOGGER_HEADER}\n{response}")
            
        return reasoning, response

class LLM(LLMBase):
    """
    LLM类，用于统一调用LLM API
    """
    def __init__(self, modelConfig:ModelConfig):
        LLMBase.__init__(self, modelConfig)
        ILLM.__init__(self)
        self.llm: LLMBase = None
        if self.modelConfig.apiType == APIType.OPENAI:
            self.llm = OpenAILLM(modelConfig)
        elif self.modelConfig.apiType == APIType.ANTHROPIC:
            self.llm = AnthropicLLM(modelConfig)
        elif self.modelConfig.apiType == APIType.QWEN:
            self.llm = QwenLLM(modelConfig)
        elif self.modelConfig.apiType == APIType.DEEPSEEK:
            self.llm = DeepSeekReasoningLLM(modelConfig)
        elif self.modelConfig.apiType == APIType.MOCK:
            self.llm = MockLLM(modelConfig)
        else:
            raise ValueError(f"不支持的API类型: {self.modelConfig.apiType}")


    def call(self, system_role: str, user_content: str, 
             temperature:float=0.3, max_tokens:int=4096, 
             response_in_json:bool=False, log_output:bool=False) -> str:
        return self.llm.call(system_role, user_content, 
                             temperature, max_tokens,
                             response_in_json=response_in_json, 
                             log_output=log_output
                            )

    def callReasoning(self, system_role: str, user_content: str, 
                      temperature:float=0.3, max_tokens:int=4096, 
                      response_in_json:bool=False, 
                      log_output:bool=False) -> tuple[str, str]:
        return self.llm.callReasoning(system_role, user_content, 
                                      temperature, max_tokens,
                                      response_in_json=response_in_json, 
                                      log_output=log_output
                                     )