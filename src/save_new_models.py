import logging
import argparse
import mysql.connector
from common.config import load_config, get_api_key_secret_key, get_db_config
from common.llm_api import ModelConfig
from common.api_key_encryptor import ApiKeyEncryptor
from common.llm_api import ModelConfig, load_models_configs

config = load_config()

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

apiKeyEncryptor:ApiKeyEncryptor = ApiKeyEncryptor(get_api_key_secret_key(config))
db_config = get_db_config(config)

def save_new_models_to_db(model_configs: list[ModelConfig], db_config: dict, encryptor: ApiKeyEncryptor):
    """将新模型配置保存到数据库的通用方法
    
    Args:
        model_configs: 要保存的模型配置列表
        db_config: 数据库配置
        encryptor: API密钥加密器
    
    Returns:
        int: 成功保存的模型数量
    """
    if not model_configs:
        print("没有需要保存的模型配置")
        return 0
    
    conn = mysql.connector.connect(**db_config)
    saved_models = []  # 保存成功的模型配置列表
    
    try:
        with conn.cursor() as cursor:
            insert_sql = """
                INSERT INTO llm_models (
                    model_name, model_id, model_remark, base_url,
                    encrypted_api_key, api_type, model_type
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            
            saved_count = 0
            for model in model_configs:
                try:
                    encrypted_key = encryptor.encrypt(model.api_key)
                    cursor.execute(insert_sql, (
                        model.model_name,
                        model.model_id,
                        model.model_remark,
                        model.base_url,
                        encrypted_key,
                        model.apiType.value,
                        model.modelType.value
                    ))
                    
                    # 获取新插入记录的ID
                    new_id = cursor.lastrowid
                    # 创建带有数据库ID的新模型配置
                    saved_model = ModelConfig(
                        id=new_id,
                        model_name=model.model_name,
                        model_id=model.model_id,
                        api_key=model.api_key,
                        base_url=model.base_url,
                        model_remark=model.model_remark,
                        api_type=model.apiType.value,
                        model_type=model.modelType.value,
                        new_model=True
                    )
                    saved_models.append(saved_model)
                    
                    saved_count += 1
                    print(f"成功保存模型配置: {model.model_name}")
                except mysql.connector.Error as e:
                    print(f"保存模型配置失败 {model.model_name}: {e}")
            
            conn.commit()
            print(f"总共成功保存 {saved_count} 个模型配置")
            
            # 输出新增模型信息
            if saved_models:
                print(ModelConfig.get_new_models_info(saved_models))
            
            return saved_count
    
    except mysql.connector.Error as e:
        print(f"数据库操作失败: {e}")
        conn.rollback()
        return 0
    finally:
        conn.close()

def print_encrypted_api_key(api_key):
    print(apiKeyEncryptor.encrypt(api_key))

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='模型配置工具')
    parser.add_argument('--mode', type=str, choices=['save', 'encrypt'], default='save',
                        help='运行模式: save - 保存新模型配置到数据库, encrypt - 加密API密钥')
    parser.add_argument('--file', type=str, default='new_models_to_save.json',
                        help='模型配置文件路径 (仅在save模式下使用)')
    parser.add_argument('--api-key', type=str, help='需要加密的API密钥明文 (仅在encrypt模式下使用)')
    
    args = parser.parse_args()
    
    if args.mode == 'save':
        # 加载所有模型配置
        all_models = load_models_configs(
            apiKeyEncryptor=apiKeyEncryptor,
            file_path=args.file,
            db_config=db_config
        )
        
        # 筛选出new_model=true的配置
        new_models = []
        for model in all_models:
            # 检查原始配置中是否标记为new_model
            if model.is_new_model():
                new_models.append(model)
        
        if not new_models:
            print("没有找到标记为new_model=true的模型配置")
            return
        
        print(f"找到 {len(new_models)} 个新模型配置:")
        for model in new_models:
            print(f"  - {model.model_name} ({model.model_id})")
        
        # 保存到数据库
        saved_count = save_new_models_to_db(new_models, db_config, apiKeyEncryptor)
        print(f"操作完成，成功保存 {saved_count} 个模型配置")
    
    elif args.mode == 'encrypt':
        if not args.api_key:
            parser.error('在encrypt模式下，--api-key参数是必需的')
        print_encrypted_api_key(args.api_key)

if __name__ == '__main__':
    main()