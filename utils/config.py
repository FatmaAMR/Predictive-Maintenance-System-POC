from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str
    APP_VERSION: str
    OPENAI_API_KEY: str
    OPENAI_MODEL_NAME: str
    OPENAI_API_BASE: str
    class Config:
        env_file = ".env"
        extra = "ignore"
        
def get_settings(): 
    return Settings()