"""
데이터 로딩 및 관리 모듈
"""
import pandas as pd
import streamlit as st
from typing import Dict, Any
from config.settings import Config


class DataLoader:
    """데이터 로딩 클래스"""
    
    def __init__(self):
        self.config = Config()
    
    @st.cache_data
    def load_main_excel_data(_self) -> pd.DataFrame:
        """메인 약물 데이터 로딩"""
        try:
            df = pd.read_excel(_self.config.MAIN_DATA_FILE)
            # 필요한 컬럼들의 데이터 타입 정리
            for col in ['제품명', '업체명', '식별표기']:
                if col in df.columns:
                    df[col] = df[col].astype(str)
            return df
        except FileNotFoundError:
            st.error(f"데이터 파일을 찾을 수 없습니다: {_self.config.MAIN_DATA_FILE}")
            return pd.DataFrame()
    
    @st.cache_data
    def load_rag_excel_data(_self) -> pd.DataFrame:
        """RAG용 약물 데이터 로딩"""
        try:
            return pd.read_excel(_self.config.RAG_DATA_FILE)
        except FileNotFoundError:
            st.error(f"RAG 데이터 파일을 찾을 수 없습니다: {_self.config.RAG_DATA_FILE}")
            return pd.DataFrame()
    
    @st.cache_data
    def load_pregnancy_warning_data(_self) -> pd.DataFrame:
        """임산부 금기 약물 데이터 로딩"""
        try:
            df = pd.read_excel(_self.config.PREGNANCY_WARNING_FILE)
            df['제품명'] = df['제품명'].str.strip()
            return df
        except FileNotFoundError:
            st.error(f"임산부 금기 데이터 파일을 찾을 수 없습니다: {_self.config.PREGNANCY_WARNING_FILE}")
            return pd.DataFrame()


class SessionManager:
    """세션 상태 관리 클래스"""
    
    @staticmethod
    def initialize_session_state():
        """세션 상태 초기화"""
        default_states = {
            'basket_01': {},
            'displayed_candidates_01': {},
            'displayed_candidates': {},  # 캘린더 페이지용 추가
            'user_question': "",
            'basket': {},
            'question_history': [],
            'last_effect': "",
            'calendar_done': False,
            'ocr_texts_01': [],
            'creds': None,  # 구글 인증 정보
            'service': None  # 구글 캘린더 서비스
        }
        
        for key, default_value in default_states.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
    
    @staticmethod
    def clear_session_state():
        """세션 상태 초기화"""
        for key in list(st.session_state.keys()):
            if key.startswith(('basket', 'displayed_candidates', 'ocr_texts')):
                del st.session_state[key]