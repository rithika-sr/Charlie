"""Encryption utilities"""
from cryptography.fernet import Fernet
import os

class DataEncryption:
    def __init__(self):
        key = os.getenv('ENCRYPTION_KEY', Fernet.generate_key().decode())
        self.cipher = Fernet(key.encode() if isinstance(key, str) else key)
    
    def encrypt(self, data: str) -> str:
        return self.cipher.encrypt(data.encode()).decode()
    
    def decrypt(self, encrypted: str) -> str:
        return self.cipher.decrypt(encrypted.encode()).decode()

encryptor = DataEncryption()
