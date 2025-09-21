"""
설정 관리 모듈
"""
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """애플리케이션 설정"""
    
    # API Keys
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
    
    # Google Calendar
    CALENDAR_SCOPES = ['https://www.googleapis.com/auth/calendar.events']
    CREDENTIALS_FILE = 'credentials.json'
    TOKEN_FILE = 'token.json'
    
    # Data Files
    MAIN_DATA_FILE = "data/final_data.xlsx"
    RAG_DATA_FILE = "data/txta.xlsx"
    PREGNANCY_WARNING_FILE = "data/pregnant_warning.xlsx"
    
    # OCR Settings
    OCR_LANGUAGES = ['en', 'ko']
    
    # AI Settings
    DEFAULT_MODEL = "gpt-4o-mini"
    DEFAULT_TEMPERATURE = 0.3
    
    # Directory Settings
    PERSIST_DIR = "./index_store"
    IMAGES_DIR = "images"
    
    @classmethod
    def validate_config(cls):
        """설정 유효성 검사"""
        required_keys = [cls.OPENAI_API_KEY]
        missing_keys = [key for key in required_keys if not key]
        
        if missing_keys:
            raise ValueError(f"Missing required environment variables: {missing_keys}")
        
        return True