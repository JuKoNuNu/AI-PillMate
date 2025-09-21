"""
PillMate - AI 기반 약물 관리 시스템
메인 애플리케이션 파일 (개선된 구조)
"""
import streamlit as st
from streamlit_option_menu import option_menu

# 로컬 모듈 임포트 (개선된 구조)
from config.settings import Config
from utils.data_loader import SessionManager
from ui.pages import MainPage, OCRPage, ChatbotPage, CalendarPage


def setup_page_config():
    """페이지 설정"""
    st.set_page_config(
        page_title="PillMate - AI 기반 약물 관리 시스템",
        layout="wide",
        initial_sidebar_state="expanded"
    )


def create_sidebar_menu():
    """사이드바 메뉴 생성"""
    with st.sidebar:
        page = option_menu(
            "MENU",
            ["소개", "알약 이미지 검색", "약 정보 검색 챗봇", "복약 캘린더"],
            icons=["capsule", "camera", "robot", "calendar"],
            default_index=0,
            styles={
                "container": {
                    "padding": "5px", 
                    "background-color": "#f0f0f0"
                },
                "icon": {
                    "color": "black", 
                    "font-size": "20px"
                },
                "nav-link": {
                    "font-size": "16px", 
                    "text-align": "left", 
                    "margin": "0px"
                },
                "nav-link-selected": {
                    "background-color": "#9a9a9a", 
                    "color": "white"
                },
            }
        )
    return page


def route_to_page(page: str):
    """페이지 라우팅 (개선된 구조)"""
    try:
        # 페이지별 인스턴스 생성 - 각각 별도 파일에서 관리
        page_instances = {
            "소개": MainPage,
            "알약 이미지 검색": OCRPage,
            "약 정보 검색 챗봇": ChatbotPage,
            "복약 캘린더": CalendarPage
        }
        
        if page in page_instances:
            if page == "소개":
                page_instances[page].show()  # 정적 메서드
            else:
                page_instance = page_instances[page]()  # 인스턴스 생성
                page_instance.show()
        
    except Exception as e:
        st.error(f"페이지 로딩 중 오류가 발생했습니다: {e}")
        st.info("페이지를 새로고침 하거나 다른 페이지를 선택해 보세요.")


def main():
    """메인 애플리케이션"""
    try:
        # 설정 검증
        Config.validate_config()
        
        # 페이지 설정
        setup_page_config()
        
        # 세션 초기화
        SessionManager.initialize_session_state()
        
        # 사이드바 메뉴
        selected_page = create_sidebar_menu()
        
        # 페이지 라우팅
        route_to_page(selected_page)
        
    except ValueError as e:
        st.error(f"설정 오류: {e}")
        st.info("환경 변수를 확인해 주세요.")
    except Exception as e:
        st.error(f"애플리케이션 시작 중 오류: {e}")
        st.info("새로고침 후 다시 시도해 주세요.")


if __name__ == "__main__":
    main()