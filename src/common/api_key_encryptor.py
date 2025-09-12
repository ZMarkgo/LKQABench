from cryptography.fernet import Fernet

class ApiKeyEncryptor:
    def __init__(self, secret_key: bytes):
        """
        初始化加解密器
        :param secret_key: 必须是一个通过 Fernet.generate_key() 生成的密钥
        """
        self.fernet = Fernet(secret_key)

    def encrypt(self, plain_text: str) -> str:
        """
        加密明文API Key
        :param plain_text: 明文API Key
        :return: 加密后的字符串（Base64编码）
        """
        encrypted = self.fernet.encrypt(plain_text.encode())
        return encrypted.decode()

    def decrypt(self, encrypted_text: str) -> str:
        """
        解密加密后的API Key
        :param encrypted_text: 加密后的字符串
        :return: 解密后的明文API Key
        """
        decrypted = self.fernet.decrypt(encrypted_text.encode())
        return decrypted.decode()

    @staticmethod
    def generate_key() -> bytes:
        """
        生成新的加解密密钥
        :return: 密钥bytes
        """
        return Fernet.generate_key()

if __name__ == "__main__":
    # 生成密钥
    secret_key = ApiKeyEncryptor.generate_key()
    print(f"Generated Secret Key: {secret_key}")

    # 初始化加解密器
    encryptor = ApiKeyEncryptor(secret_key)

    # 加密API Key: 模拟一个较长的API Key
    plain_api_key = "sk-KDAsdfGe8yy4feM4cSS8wN32sdYJHJ1fF4954sdfsdfD7fA7589UID1aB8g33B4"
    encrypted_api_key = encryptor.encrypt(plain_api_key)
    print(f"Encrypted API Key: {encrypted_api_key}")

    # 解密API Key
    decrypted_api_key = encryptor.decrypt(encrypted_api_key)
    print(f"Decrypted API Key: {decrypted_api_key}")