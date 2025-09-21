"""
OCR 기반 약물 인식 페이지
"""
import streamlit as st
from typing import List, Dict, Any
from copy import deepcopy
from PIL import Image

from core.ocr_processor import OCRProcessor
from core.drug_matcher import DrugMatcher, DrugInteractionChecker, DrugQuestionAnswerer
from utils.data_loader import DataLoader, SessionManager
from utils.text_processor import QuestionClassifier
from ..components import UIComponents, TabManager, MessageTemplates


class OCRPage:
    """OCR 기반 약물 인식 페이지"""
    
    def __init__(self):
        self.data_loader = DataLoader()
        self.ocr_processor = OCRProcessor()
        self.question_classifier = QuestionClassifier()
        self.ui = UIComponents()
        
        # 데이터 로딩
        self.drug_data = self.data_loader.load_main_excel_data()
        self.pregnancy_data = self.data_loader.load_pregnancy_warning_data()
        
        # 매처 초기화
        self.drug_matcher = DrugMatcher(self.drug_data)
        self.interaction_checker = DrugInteractionChecker(self.pregnancy_data)
        self.question_answerer = DrugQuestionAnswerer()
    
    def show(self):
        """OCR 페이지 표시"""
        st.subheader("약 이미지 인식 기반 정보 제공 시스템")
        
        # 세션 초기화
        SessionManager.initialize_session_state()
        
        # 탭 생성
        ocr_tab, select_tab, question_tab, result_tab = TabManager.create_ocr_tabs()
        
        with ocr_tab:
            self._show_image_upload_tab()
        
        with select_tab:
            self._show_drug_selection_tab()
        
        with question_tab:
            self._show_question_tab()
        
        with result_tab:
            self._show_result_tab()
    
    def _show_image_upload_tab(self):
        """이미지 업로드 탭"""
        st.header("📷 약 이미지 업로드")
        
        uploaded_files = self.ui.show_image_upload_section("01")
        st.session_state['ocr_texts_01'] = []
        
        if uploaded_files:
            st.markdown(MessageTemplates.get_ocr_guide_message())
            
            for i, uploaded_file in enumerate(uploaded_files):
                image = Image.open(uploaded_file).convert("RGB")
                cropped = self.ocr_processor.crop_pill_area(image)
                
                self.ui.show_processed_image(cropped, f"이미지 {i+1} 크롭 약 이미지")
                
                ocr_text = self.ocr_processor.extract_text(cropped).upper()
                edited = self.ui.show_ocr_result(ocr_text, i, "01")
                st.session_state['ocr_texts_01'].append(edited)
    
    def _show_drug_selection_tab(self):
        """약물 선택 탭"""
        st.header("💊 인식된 약의 정보를 확인해주세요")
        
        displayed_candidates_01 = {}
        
        if st.session_state['ocr_texts_01']:
            candidates = self.drug_matcher.find_candidates_by_identifiers(
                st.session_state['ocr_texts_01']
            )
            
            if not candidates.empty:
                self.ui.show_drug_selection_guide()
                
                for display_idx, (_, row) in enumerate(candidates.iterrows()):
                    key_cb = f"select_01_{display_idx}"
                    key_qty = f"qty_01_{display_idx}"
                    displayed_candidates_01[key_cb] = row
                    
                    with st.container():
                        cols = st.columns([1, 5])
                        with cols[1]:
                            selected = st.checkbox(
                                f"{row['식별표기']} - {row['제품명']} ({row['업체명']})", 
                                key=key_cb
                            )
                            
                            if selected:
                                qty = st.number_input(
                                    f"▶ {row['제품명']} 수량 선택", 
                                    min_value=1, max_value=10, value=1, key=key_qty
                                )
                                st.session_state['basket_01'][key_cb] = {
                                    "quantity": qty, **row.to_dict()
                                }
                            else:
                                st.session_state['basket_01'].pop(key_cb, None)
                            
                            with st.expander(" ▼ 추가 정보 보기"):
                                st.markdown(f"- **제형**: {row.get('제형', '정보 없음')}")
                                st.markdown(f"- **모양**: {row.get('모양', '정보 없음')}")
                                st.markdown(f"- **색깔**: {row.get('색깔', '정보 없음')}")
                                st.markdown(f"- **성상**: {row.get('성상', '정보 없음')}")
            else:
                self.ui.show_warning_message("조건에 맞는 약 후보가 없습니다.")
        
        st.session_state['displayed_candidates_01'] = displayed_candidates_01
        
        if not displayed_candidates_01:
            self.ui.show_info_message(MessageTemplates.get_no_candidates_message())
    
    def _show_question_tab(self):
        """질문 입력 탭"""
        st.header("❓ 궁금한 점을 입력해주세요")
        st.markdown(MessageTemplates.get_question_example_message())
        
        st.session_state['user_question'] = self.ui.show_question_input("질문 입력")
        self.ui.show_tab_navigation_guide(
            "질문을 모두 작성", 
            "④ 복용 가능 여부 결과"
        )
    
    def _show_result_tab(self):
        """결과 표시 탭"""
        st.header("✅ 복용 가능 여부 결과")
        
        displayed_candidates_01 = st.session_state.get('displayed_candidates_01', {})
        basket_items_01 = list(st.session_state['basket_01'].values())
        
        if basket_items_01 and st.session_state.get('user_question'):
            question = st.session_state['user_question']
            pill_list = [
                deepcopy(item) for key, item in st.session_state['basket_01'].items() 
                if key in displayed_candidates_01
            ]
            
            if self.question_classifier.is_pregnancy_question(question):
                result = self.interaction_checker.check_pregnancy_warnings(pill_list)
                st.markdown(result)
            elif self.question_classifier.is_combination_question(question):
                result = self.interaction_checker.check_drug_interaction_summary(pill_list)
                st.markdown(result)
            else:
                for i, pill_info in enumerate(pill_list):
                    st.subheader(f"약 {i+1}: {pill_info['제품명']}")
                    answer = self.question_answerer.chat_with_pill_info(pill_info, question)
                    st.markdown(answer)
        else:
            self.ui.show_info_message(MessageTemplates.get_select_drug_first_message())
        
        self.ui.show_question_history(st.session_state.get('question_history', []))