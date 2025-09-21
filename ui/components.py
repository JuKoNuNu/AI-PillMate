"""
UI 컴포넌트 모듈 - 완전버전
"""
import streamlit as st
from typing import List, Dict, Any
from PIL import Image
import datetime


class UIComponents:
    """공통 UI 컴포넌트 클래스"""
    
    @staticmethod
    def show_page_header(title: str, subtitle: str = ""):
        """페이지 헤더 표시"""
        st.markdown(f"""
            <h1 style='text-align: center; margin-bottom: 30px;'>{title}</h1>
        """, unsafe_allow_html=True)
        
        if subtitle:
            st.markdown(f"""
                <h3 style='text-align: center; color: #666; margin-bottom: 40px;'>{subtitle}</h3>
            """, unsafe_allow_html=True)
    
    @staticmethod
    def show_info_box(content: str, box_type: str = "info"):
        """정보 박스 표시"""
        colors = {
            "info": "#e1f5fe",
            "warning": "#fff3e0", 
            "error": "#ffebee",
            "success": "#e8f5e8"
        }
        
        color = colors.get(box_type, colors["info"])
        
        st.markdown(f"""
            <div style='
                margin-top: 30px; border: 2px solid #ccc; border-radius: 10px; 
                padding: 20px; background-color: {color};
            '>
                {content}
            </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def show_drug_candidate(row: Dict[str, Any], display_idx: int, prefix: str = "") -> tuple:
        """약물 후보 표시 및 선택"""
        key_cb = f"select_{prefix}{display_idx}"
        key_qty = f"qty_{prefix}{display_idx}"
        
        selected = st.checkbox(
            f"{row.get('식별표기', 'N/A')} - {row.get('제품명', 'N/A')} ({row.get('업체명', 'N/A')})", 
            key=key_cb
        )
        
        quantity = 1
        if selected:
            quantity = st.number_input(
                f"▶ {row.get('제품명', 'N/A')} 수량 선택", 
                min_value=1, max_value=10, value=1, key=key_qty
            )
        
        # 추가 정보 표시
        with st.expander(" ▼ 추가 정보 보기"):
            st.markdown(f"- **제형**: {row.get('제형', '정보 없음')}")
            st.markdown(f"- **모양**: {row.get('모양', '정보 없음')}")
            st.markdown(f"- **색깔**: {row.get('색깔', '정보 없음')}")
            st.markdown(f"- **성상**: {row.get('성상', '정보 없음')}")
            
            # 제품 이미지 표시
            img_url = row.get("제품이미지")
            if isinstance(img_url, str) and img_url.startswith("http"):
                st.image(img_url, width=150)
        
        st.markdown("---")
        
        return selected, quantity, key_cb, key_qty
    
    @staticmethod
    def show_image_upload_section(key_suffix: str = "") -> List:
        """이미지 업로드 섹션"""
        st.header("📷 약 이미지 업로드")
        
        uploaded_files = st.file_uploader(
            "궁금하신 약의 이미지를 업로드 해주세요", 
            type=["jpg", "jpeg", "png"], 
            accept_multiple_files=True,
            key=f"file_upload_{key_suffix}"
        )
        
        return uploaded_files or []
    
    @staticmethod
    def show_processed_image(image: Image.Image, caption: str, width: int = 200):
        """처리된 이미지 표시"""
        st.image(image, caption=caption, width=width)
    
    @staticmethod
    def show_ocr_result(text: str, index: int, key_suffix: str = "") -> str:
        """OCR 결과 표시 및 수정 가능한 입력"""
        return st.text_input(
            f"약 {index+1} 식별표기 (수정 가능)", 
            value=text, 
            key=f"ocr_edit_{key_suffix}_{index}"
        )
    
    @staticmethod
    def show_drug_selection_guide():
        """약물 선택 가이드 표시"""
        st.markdown("""
            📝 아래 목록 중 **복용한 약을 선택**하고 수량을 입력해 주세요.
        """)
    
    @staticmethod
    def show_question_input(placeholder: str = "질문을 입력해주세요", key: str = "question") -> str:
        """질문 입력 섹션"""
        return st.text_input(placeholder, key=key)
    
    @staticmethod
    def show_question_textarea(placeholder: str = "질문을 입력해주세요", height: int = 100, key: str = "question_area") -> str:
        """질문 입력 텍스트 영역"""
        return st.text_area(placeholder, height=height, key=key)
    
    @staticmethod
    def show_calendar_controls():
        """캘린더 제어 섹션"""
        col1, col2 = st.columns(2)
        
        with col1:
            dosage_times = st.multiselect(
                "복약 시간 선택", 
                ["05:00", "06:00", "07:00", "08:00", "09:00", "10:00", "11:00", "12:00",
                 "13:00", "14:00", "15:00", "16:00", "17:00", "18:00", "19:00", "20:00",
                 "21:00", "22:00", "23:00"]
            )
        
        with col2:
            start_date = st.date_input("시작 날짜", datetime.date.today())
            end_date = st.date_input("종료 날짜", datetime.date.today())
        
        return dosage_times, start_date, end_date
    
    @staticmethod
    def show_calendar_buttons():
        """캘린더 관련 버튼들"""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            retry_login = st.button("🔐 로그인 다시 시도")
        
        with col2:
            register_schedule = st.button("📅 복약 일정 등록")
        
        with col3:
            delete_schedule = st.button("🗑️ 복약 일정 삭제하기")
        
        return retry_login, register_schedule, delete_schedule
    
    @staticmethod
    def show_loading_spinner(text: str = "처리 중..."):
        """로딩 스피너 표시"""
        return st.spinner(text)
    
    @staticmethod
    def show_success_message(message: str):
        """성공 메시지 표시"""
        st.success(message)
    
    @staticmethod
    def show_error_message(message: str):
        """에러 메시지 표시"""
        st.error(message)
    
    @staticmethod
    def show_warning_message(message: str):
        """경고 메시지 표시"""
        st.warning(message)
    
    @staticmethod
    def show_info_message(message: str):
        """정보 메시지 표시"""
        st.info(message)
    
    @staticmethod
    def show_ai_consultation_button() -> bool:
        """AI 상담 버튼"""
        return st.button("💬 AI 질문하기")
    
    @staticmethod
    def show_question_history(question_history: List[Dict[str, str]]):
        """이전 질문 기록 표시"""
        if question_history:
            with st.expander("📝 이전 질문 보기"):
                for i, qa in enumerate(reversed(question_history), 1):
                    st.markdown(f"**{i}. 질문:** {qa['question']}")
                    st.markdown(f"**답변:** {qa['answer']}")
                    st.markdown("---")
    
    @staticmethod
    def show_tab_navigation_guide(current_tab: str, next_tab: str):
        """탭 네비게이션 가이드"""
        st.markdown(f"✏️ {current_tab} 완료하셨나요? 그렇다면 👉 '{next_tab}' 탭으로 이동해 주세요.")
    
    @staticmethod
    def show_result_summary(title: str):
        """결과 요약 헤더"""
        st.header(title)
        st.markdown("---")


class TabManager:
    """탭 관리 클래스"""
    
    @staticmethod
    def create_tabs(tab_names: List[str]):
        """탭 생성"""
        return st.tabs(tab_names)
    
    @staticmethod
    def create_ocr_tabs():
        """OCR 페이지용 탭 생성"""
        return st.tabs([
            "① 약 이미지 업로드",
            "② 인식 결과 & 약 선택", 
            "③ 질문 입력",
            "④ 복용 가능 여부 결과"
        ])


class MessageTemplates:
    """메시지 템플릿 클래스"""
    
    @staticmethod
    def get_ocr_guide_message():
        """OCR 가이드 메시지"""
        return "🔍 인식된 식별표기를 확인 후 **본인의 약과 일치하면 '② 인식 결과 & 약 선택' 탭으로 넘어가세요.**"
    
    @staticmethod
    def get_question_example_message():
        """질문 예시 메시지"""
        return "예: '이 약들을 같이 먹어도 되나요?', '임산부가 복용해도 될까요?' 등"
    
    @staticmethod
    def get_consultation_example_message():
        """상담 예시 메시지"""
        return "궁금한 점을 입력해 주세요 (예: 같이 먹어도 되나요?, 보관 방법은?)"
    
    @staticmethod
    def get_no_candidates_message():
        """후보 없음 메시지"""
        return "⚠️ 인식된 약이 없거나 선택하지 않았습니다. 이미지 또는 OCR 결과를 다시 확인해 주세요."
    
    @staticmethod
    def get_select_drug_first_message():
        """약물 선택 먼저 메시지"""
        return "💬 먼저 약을 선택하고 질문을 입력해주세요."