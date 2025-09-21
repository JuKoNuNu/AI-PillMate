"""
메인 소개 페이지
"""
import streamlit as st
from ..components import UIComponents


class MainPage:
    """메인 소개 페이지"""
    
    @staticmethod
    def show():
        """메인 페이지 표시"""
        st.markdown("""
            <h1 style='text-align: center; margin-bottom: 60px;'> PillMate 🤒 </h1>
            <h2 style='text-align: left; margin-top: 60px;'>안녕하세요. 비대면 개인 건강 비서 PillMate 입니다!</h2>
        """, unsafe_allow_html=True)

        UIComponents.show_info_box("""
            💊 저희는 AI 기반 다제약물(중복 복용) 예방을 목적으로 다음과 같은 기능을 제공합니다: <br><br>
            📷 <b>알약 이미지 검색</b>: 사진을 업로드하면 알약을 인식하여 복용 중복 가능 정보를 확인할 수 있어요.<br><br>
            🤖 <b>약 정보 검색 챗봇</b>: 궁금한 약 정보 질문을 하고, AI로부터 답변을 받아보세요.<br><br>
            💡 <b>복약 캘린더</b>: 캘린더를 활용해 약 복용 시간을 기록할 수 있어요.<br><br>
        """)

        st.markdown("<p style='margin-top: 30px;'>▶ 왼쪽 사이드바에서 기능을 선택해주세요!</p>", 
                   unsafe_allow_html=True)