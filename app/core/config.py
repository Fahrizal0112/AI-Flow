from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    PROJECT_NAME: str = "AI Generator API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    POSTGRES_SERVER: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    
    REDIS_URL: str = "redis://localhost:6379/0"
    
    MLFLOW_TRACKING_URI: str = "http://localhost:5001"
    
    MODEL_STORAGE_PATH: str = "models"
    
    # Deepseek API settings
    DEEPSEEK_API_KEY: str
    DEEPSEEK_API_URL: str = "https://api.deepseek.com/v1/chat/completions"
    
    # Gemini API settings
    GEMINI_API_KEY: str
    
    class Config:
        env_file = ".env"

settings = Settings() 