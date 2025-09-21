"""
복약 캘린더 페이지
"""
import os
import pandas as pd
import streamlit as st
from datetime import date
from typing import List
from PIL import Image

from core.ocr_processor import OCRProcessor
from core.drug_matcher import DrugMatcher, DrugInteractionChecker, DrugQuestionAnswerer
from services.calendar_service import CalendarService
from services.drug_service import DrugInformationService
from utils.data_loader import DataLoader, SessionManager
from utils.text_processor import QuestionClassifier
from ..components import UIComponents


class CalendarPage:
    """복약 캘린더 페이지"""
    
    def __init__(self):
        self.data_loader = DataLoader()
        self.ocr_processor = OCRProcessor()
        self.calendar_service = CalendarService()
        self.ui = UIComponents()
        
        # 데이터 로딩
        self.drug_data = self.data_loader.load_main_excel_data()
        self.drug_matcher = DrugMatcher(self.drug_data)
        self.drug_service = DrugInformationService(self.data_loader.load_rag_excel_data())
        
        # 세션 초기화
        SessionManager.initialize_session_state()
    
    def show(self):
        """캘린더 페이지 표시"""
        st.subheader("약 이미지 인식 기반 캘린더 등록")
        
        # 이미지 업로드 및 처리
        self._show_image_processing_section()
        
        # 캘린더 기능
        self._show_calendar_section()
        
        # AI 상담 기능
        self._show_consultation_section()
    
    def _show_image_processing_section(self):
        """이미지 처리 섹션"""
        uploaded_files = self.ui.show_image_upload_section("calendar")
        
        if uploaded_files:
            st.header("업로드한 약 식별표기 확인")
            
            # OCR 처리
            edited_texts = []
            for i, uploaded_file in enumerate(uploaded_files):
                image = Image.open(uploaded_file).convert("RGB")
                cropped = self.ocr_processor.crop_pill_area(image)
                
                self.ui.show_processed_image(cropped, f"이미지 {i+1} 크롭 약 이미지")
                
                ocr_text = self.ocr_processor.extract_text(cropped).upper()
                edited = self.ui.show_ocr_result(
                    ocr_text, i, f"calendar_이미지 {i+1} 식별표기 (이미지 인식이 잘못 되었다면 수정해주세요)"
                )
                edited_texts.append(edited)
            
            # 약물 후보 검색 및 선택
            self._show_drug_candidates(edited_texts)
    
    def _show_drug_candidates(self, edited_texts: List[str]):
        """약물 후보 표시 및 선택"""
        st.header("식별표기 확인 후 약을 선택하고 개수를 지정하세요")
        
        candidates = self.drug_matcher.find_candidates_by_identifiers(edited_texts)
        
        if candidates.empty:
            self.ui.show_warning_message("조건에 맞는 약 후보가 없습니다.")
            return

        # 원본 로직: displayed_candidates를 세션에 저장
        displayed_candidates = {}

        for display_idx, (_, row) in enumerate(candidates.iterrows()):
            key_cb = f"select_{display_idx}"
            key_qty = f"qty_{display_idx}"

            selected = st.checkbox(
                f"{row['식별표기']} - {row['제품명']} ({row['업체명']})", 
                key=key_cb, 
                value=key_cb in st.session_state['basket']
            )

            displayed_candidates[key_cb] = row

            if selected:
                qty = st.number_input(f"▶ {row['제품명']} 수량 선택", min_value=1, max_value=10, value=1, key=key_qty)
                st.session_state['basket'][key_cb] = {"quantity": qty}
            else:
                st.session_state['basket'].pop(key_cb, None)

            with st.expander(" ▼ 추가 정보 보기"):
                st.markdown(f"- **제형**: {row.get('제형', '정보 없음')}")
                st.markdown(f"- **모양**: {row.get('모양', '정보 없음')}")
                st.markdown(f"- **색깔**: {row.get('색깔', '정보 없음')}")
                st.markdown(f"- **성상**: {row.get('성상', '정보 없음')}")
                img_url = row.get("제품이미지")
                if isinstance(img_url, str) and img_url.startswith("http"):
                    st.image(img_url, width=150)
            st.markdown("---")

        # 세션에 displayed_candidates 저장 (핵심!)
        st.session_state['displayed_candidates'] = displayed_candidates
    
    def _show_calendar_section(self):
        """캘린더 관련 기능"""
        st.title("💊 복약 일정 캘린더 등록")
        
        dosage_times, start_date, end_date = self.ui.show_calendar_controls()
        
        # 버튼들
        retry_login, register_schedule, delete_schedule = self.ui.show_calendar_buttons()
        
        if retry_login:
            self._retry_login()
        
        if register_schedule:
            self._register_medication_schedule(dosage_times, start_date, end_date)
        
        if delete_schedule:
            self._delete_medication_schedule(start_date, end_date)
        
        # 캘린더 표시
        if st.session_state.get("calendar_done") and "service" in st.session_state:
            self.calendar_service.show_calendar(start_date, end_date)
    
    def _retry_login(self):
        """로그인 재시도"""
        if os.path.exists("token.json"):
            os.remove("token.json")
            self.ui.show_success_message("✅ 기존 로그인 정보 삭제 완료. 다시 로그인 해주세요.")
    
    def _register_medication_schedule(self, dosage_times: List[str], start_date: date, end_date: date):
        """복약 일정 등록"""
        if not dosage_times:
            self.ui.show_warning_message("복약 시간을 선택해주세요.")
            return
        
        if not st.session_state['basket']:
            self.ui.show_warning_message("약을 선택해주세요.")
            return
        
        try:
            # 서비스 설정
            if "creds" not in st.session_state:
                st.session_state["creds"] = self.calendar_service.get_credentials()
            
            if "service" not in st.session_state:
                st.session_state["service"] = self.calendar_service.get_service()
            
            # 선택된 약물들 처리
            displayed_candidates = st.session_state.get('displayed_candidates', {})
            basket_keys = list(st.session_state['basket'].keys())
            
            success_count = 0
            for key in basket_keys[:5]:  # 최대 5개
                if key not in displayed_candidates:
                    continue
                
                drug_info = displayed_candidates[key]
                product_name = drug_info["제품명"]
                
                # 용법용량 정보 가져오기
                usage_instruction = self.drug_service.get_usage_instruction(product_name)
                
                # 캘린더 이벤트 생성
                if self.calendar_service.create_medication_event(
                    product_name, dosage_times, start_date, end_date, usage_instruction
                ):
                    success_count += 1
            
            if success_count > 0:
                self.ui.show_success_message("✅ 복약 일정이 Google 캘린더에 등록되었습니다.")
                st.session_state["calendar_done"] = True
            else:
                self.ui.show_error_message("❌ 복약 일정 등록에 실패했습니다.")
                
        except Exception as e:
            self.ui.show_error_message(f"❌ 캘린더 등록 중 오류: {e}")
    
    def _delete_medication_schedule(self, start_date: date, end_date: date):
        """복약 일정 삭제"""
        try:
            if "service" in st.session_state:
                deleted = self.calendar_service.delete_medication_events(start_date, end_date)
                if deleted:
                    self.ui.show_success_message(f"✅ 총 {deleted}개의 복약 이벤트를 삭제했습니다.")
                else:
                    self.ui.show_info_message("ℹ️ 삭제할 복약 이벤트가 없습니다.")
                
                # 캘린더 갱신
                self.calendar_service.show_calendar(start_date, end_date)
        except Exception as e:
            self.ui.show_error_message(f"❌ 삭제 실패: {e}")
    
    def _show_consultation_section(self):
        """AI 상담 섹션"""
        st.subheader("💬 AI 복약 상담")
        
        if "question_history" not in st.session_state:
            st.session_state["question_history"] = []
        
        question = self.ui.show_question_textarea(
            "궁금한 점을 입력해 주세요 (예: 같이 먹어도 되나요?, 보관 방법은?)"
        )
        
        if self.ui.show_ai_consultation_button():
            if not question.strip():
                self.ui.show_warning_message("질문을 입력해주세요.")
            else:
                self._handle_consultation_question(question)
    
    def _handle_consultation_question(self, question: str):
        """상담 질문 처리"""
        st.header("💬 AI 복약 상담 결과")
        
        # 원본과 동일한 방식으로 basket_items 구성
        displayed_candidates = st.session_state.get('displayed_candidates', {})
        basket_items = [displayed_candidates[key] for key in st.session_state['basket'].keys() if key in displayed_candidates]
        quantities = [st.session_state['basket'][key]["quantity"] for key in st.session_state['basket'].keys() if key in displayed_candidates]

        if not basket_items:
            self.ui.show_warning_message("먼저 약을 선택해주세요.")
            return

        question_classifier = QuestionClassifier()
        
        if question_classifier.is_combination_question(question):
            # 원본 로직: 바로 병용 분석 수행
            interaction_summary = self._check_drug_interactions(basket_items)
            dosage_check = self._check_dosage_safety(basket_items, quantities)
            
            st.markdown(interaction_summary)
            st.markdown(dosage_check)
            
            full_answer = interaction_summary + "\n\n" + dosage_check
        else:
            # 개별 약물 질문 처리
            full_answer = ""
            answerer = DrugQuestionAnswerer()
            
            for i, pill_info in enumerate(basket_items):
                st.subheader(f"약 {i+1}: {pill_info['제품명']}")
                # pill_info가 pandas Series인 경우 dict로 변환
                pill_dict = pill_info.to_dict() if hasattr(pill_info, 'to_dict') else pill_info
                answer = answerer.chat_with_pill_info(pill_dict, question)
                st.write(answer)
                full_answer += f"약 {i+1} ({pill_info['제품명']}): {answer}\n\n"
            
            # 추가 안전성 검사도 원본처럼 수행
            interaction_summary = self._check_drug_interactions(basket_items)
            dosage_check = self._check_dosage_safety(basket_items, quantities)
            
            st.markdown(interaction_summary)
            st.markdown(dosage_check)
            
            full_answer += interaction_summary + "\n\n" + dosage_check
        
        # 질문 기록 저장
        st.session_state["question_history"].append({
            "question": question,
            "answer": full_answer
        })
        
        # 이전 질문 보기
        self.ui.show_question_history(st.session_state.get("question_history", []))
    
    def _check_drug_interactions(self, basket_items: List) -> str:
        """약물 상호작용 검사"""
        checker = DrugInteractionChecker(pd.DataFrame())
        return checker.check_drug_interaction_summary(basket_items)
    
    def _check_dosage_safety(self, basket_items: List, quantities: List[int]) -> str:
        """성분 중복 및 용량 안전성 검사"""
        checker = DrugInteractionChecker(pd.DataFrame())
        pill_dicts = [item.to_dict() if hasattr(item, 'to_dict') else item for item in basket_items]
        return checker.check_ingredient_overlap_and_dosage(pill_dicts, quantities)