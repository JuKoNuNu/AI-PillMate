"""
AI 챗봇 페이지
"""
import re
import streamlit as st
from typing import Dict

from core.ai_engine import RAGEngine
from services.search_service import GoogleSearchService
from services.drug_service import DrugInformationService
from utils.data_loader import DataLoader
from utils.text_processor import QuestionClassifier
from ..components import UIComponents


class ChatbotPage:
    """AI 챗봇 페이지"""
    
    def __init__(self):
        self.data_loader = DataLoader()
        self.rag_engine = RAGEngine()
        self.search_service = GoogleSearchService()
        self.question_classifier = QuestionClassifier()
        self.ui = UIComponents()
        
        # 데이터 로딩
        self.drug_data = self.data_loader.load_rag_excel_data()
        self.drug_service = DrugInformationService(self.drug_data)
    
    def show(self):
        """챗봇 페이지 표시"""
        st.subheader("💊 AI 기반 약품 추천 챗봇")
        st.markdown("사용자 질문에 따라 약 정보를 추천합니다.")
        
        user_question = st.text_input(
            "❓ 증상이나 궁금한 약을 입력하세요\n 예시) 내가 지금 몸이 열이 나고 콧물이 나서 코가 막혀 어떤 약이 좋을까?"
        )
        
        if user_question:
            if self.question_classifier.is_small_talk(user_question):
                self._handle_small_talk()
            else:
                self._handle_drug_question(user_question)
    
    def _handle_small_talk(self):
        """일반 대화 처리"""
        st.markdown("🤖 **기본 대화 응답입니다.**")
        base_response = "안녕하세요! 무엇을 도와드릴까요?"
        self.ui.show_success_message("✅ AI 응답 완료")
        st.markdown(base_response)
    
    def _handle_drug_question(self, question: str):
        """약물 관련 질문 처리"""
        st.markdown("🤖 **LlamaIndex 기반 RAG로 답변 중...**")
        
        with self.ui.show_loading_spinner("AI가 답변을 생성하는 중..."):
            response = self.rag_engine.query(question)
        
        self.ui.show_success_message("✅ AI 응답 완료")
        
        # 응답에서 정보 추출
        full_response = response.response if hasattr(response, 'response') else str(response)
        product_info = self._extract_product_info(full_response)
        
        if product_info["product_name"] and product_info["product_name"].lower() != "알 수 없음":
            if product_info["product_name"].lower() != "xxx":
                self._display_drug_info(product_info, full_response)
                self._show_drug_images(product_info["product_name"])
                self._show_additional_info(product_info["product_name"])
            else:
                self.ui.show_warning_message("❗ 제품명을 찾지 못했습니다.")
    
    def _extract_product_info(self, response: str) -> Dict[str, str]:
        """응답에서 제품 정보 추출"""
        product_match = re.search(r"제품명\s*[:：]\s*(.+)", response)
        class_match = re.search(r"구분\s*[:：]\s*(.+)", response)
        effect_match = re.search(r"효능효과\s*[:：]\s*(.+)", response)
        
        product_name = product_match.group(1).strip() if product_match else "알 수 없음"
        product_class = class_match.group(1).strip() if class_match else ""
        effect_text = effect_match.group(1).strip() if effect_match else ""
        
        # 세션 상태 업데이트
        if effect_text:
            st.session_state["last_effect"] = effect_text
        elif not self.question_classifier.is_followup_question(response):
            st.session_state["last_effect"] = ""
        
        return {
            "product_name": product_name,
            "product_class": product_class,
            "effect_text": effect_text
        }
    
    def _display_drug_info(self, product_info: Dict[str, str], full_response: str):
        """약물 정보 표시"""
        # 답변 텍스트에서 정형화된 정보 제거
        answer_text = self._clean_response_text(full_response)
        
        st.markdown(f"### 💊 {product_info['product_name']} ({product_info['product_class']})")
        st.markdown(f"**AI 답변:** {answer_text}")
    
    def _clean_response_text(self, full_response: str) -> str:
        """응답 텍스트에서 정형화된 정보 제거"""
        answer_text = full_response
        patterns = [r"제품명\s*[:：].+", r"구분\s*[:：].+", r"효능효과\s*[:：].+"]
        for pattern in patterns:
            answer_text = re.sub(pattern, "", answer_text).strip()
        return answer_text
    
    def _show_drug_images(self, product_name: str):
        """약물 이미지 표시"""
        image_urls = self.search_service.search_drug_images(product_name)
        if image_urls:
            st.image(image_urls[0], caption=f"{product_name} (검색 이미지)", width=300)
        else:
            self.ui.show_info_message("🔍 구글 이미지 검색 결과 없음")
    
    def _show_additional_info(self, product_name: str):
        """추가 정보 표시"""
        st.markdown("추가적인 정보입니다.")
        
        drug_info = self.drug_service.get_drug_info(product_name)
        if drug_info:
            st.markdown(f"💊 {product_name} ({drug_info.get('구분', 'N/A')})")
            st.markdown(f"**효능효과:** {drug_info.get('효능효과', 'N/A')}")
            
            # 주의사항 표시
            safety_info = self.drug_service.get_safety_info(product_name)
            warning = safety_info.get("사용시주의사항", "")
            
            if warning and warning.strip() and warning != "정보 없음":
                with st.expander("📌 사용시 주의사항 보기"):
                    st.markdown(warning)
            else:
                self.ui.show_info_message("⚠️ 주의사항 정보가 없습니다.")
                