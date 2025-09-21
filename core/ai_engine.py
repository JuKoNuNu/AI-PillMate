"""
AI 엔진 모듈 - OpenAI와 LlamaIndex 처리
"""
import os
import streamlit as st
from openai import OpenAI
from llama_index.core import (
    VectorStoreIndex, Document, StorageContext, 
    load_index_from_storage, Settings
)
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from typing import Dict, Any, List
from config.settings import Config
from utils.text_processor import QuestionClassifier


class OpenAIClient:
    """OpenAI 클라이언트 클래스"""
    
    def __init__(self):
        self.config = Config()
        self.client = OpenAI(api_key=self.config.OPENAI_API_KEY)
    
    def extract_keyword(self, question: str) -> str:
        """핵심 키워드 추출"""
        messages = [
            {
                "role": "system", 
                "content": (
                    "당신은 한국어 질문에서 핵심 키워드를 정확히 한 단어로 추출하는 AI 약사입니다. "
                    "사용자의 질문에서 가장 중요한 증상 또는 약과 관련된 단어만 뽑아야 합니다. "
                    "예시:\n"
                    "'감기 걸린 것 같아' → '감기'\n"
                    "'두통이 심해요' → '두통'\n"
                    "'어지럽고 기운이 없어요' → '어지럼증'\n"
                    "'목이 따끔거리고 열이 나요' → '목감기'\n"
                    "정확히 한 단어로만 출력하세요."
                )
            },
            {
                "role": "user", 
                "content": f"질문: {question}\n핵심 키워드 한 단어만 출력해줘"
            }
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.DEFAULT_MODEL,
                messages=messages,
                temperature=0.2,
                max_tokens=10,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"키워드 추출 중 오류: {e}")
            return ""


class RAGEngine:
    """RAG (Retrieval-Augmented Generation) 엔진"""
    
    def __init__(self):
        self.config = Config()
        self.query_engine = None
    
    def load_documents_from_excel(self, filepath: str) -> List[Document]:
        """엑셀 파일에서 문서 로딩"""
        import pandas as pd
        
        try:
            df = pd.read_excel(filepath)
            documents = []
            
            for _, row in df.iterrows():
                content = f"""
                제품명: {row.get('제품명')}
                효능효과: {row.get('효능효과')}
                구분: {row.get('구분')}
                사용시주의사항: {row.get('사용시주의사항', '')}
                저장_이미지_파일명: {row.get('저장_이미지_파일명', '')}
                """
                documents.append(Document(text=content.strip()))
            
            return documents
        except Exception as e:
            print(f"문서 로딩 중 오류: {e}")
            return []
    
    @st.cache_resource(show_spinner=True)
    def get_query_engine(_self):
        """쿼리 엔진 생성 및 캐싱"""
        persist_dir = _self.config.PERSIST_DIR
        
        # 인덱스가 없으면 새로 생성
        if not os.path.exists(persist_dir):
            docs = _self.load_documents_from_excel(_self.config.RAG_DATA_FILE)
            if not docs:
                raise ValueError("문서를 로드할 수 없습니다.")
            
            index = VectorStoreIndex.from_documents(docs)
            index.storage_context.persist(persist_dir=persist_dir)
        else:
            # 기존 인덱스 로드
            storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
            index = load_index_from_storage(storage_context)
        
        # LLM 설정
        llm = LlamaOpenAI(
            model=_self.config.DEFAULT_MODEL,
            temperature=_self.config.DEFAULT_TEMPERATURE,
            system_prompt=(
                "당신은 한국어로만 대답하는 약사입니다. 사용자의 질문에 따라 적절한 약 하나를 친절하고 상세하게 설명합니다. "
                "예시: 내가 지금 몸이 열이 나고 콧물이 나서 코가 막혀 어떤 약이 좋을까? -> "
                "몸에 열이 나고 콧물이 나서 코가 막히는 건 감기 증상입니다. 타이레놀을 추천드립니다. "
                "이것은 일반의약품입니다. 효능은 해열,감기,진통에 도움을 줍니다. 주의사항으로는 XXX가 있습니다.\n\n"
                "한 번에 하나의 제품만 추천하되, 사용자가 '다른 약 추천해줘' 등으로 후속 질문을 하면, "
                "같은 증상에 대해 다른 제품을 추천해 주세요.\n\n"
                "설명 이후 다음 형식으로 정보를 제공합니다:\n"
                "제품명: XXX\n구분: XXX\n효능효과: XXX\n사용시주의사항: XXX"
            )
        )
        Settings.llm = llm
        
        # 쿼리 엔진 구성
        retriever = VectorIndexRetriever(index=index, similarity_top_k=2)
        response_synthesizer = get_response_synthesizer(response_mode="tree_summarize")
        
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer
        )
        
        return query_engine
    
    def query(self, question: str) -> str:
        """질문에 대한 응답 생성"""
        if not self.query_engine:
            self.query_engine = self.get_query_engine()
        
        try:
            response = self.query_engine.query(question)
            return response.response
        except Exception as e:
            print(f"RAG 쿼리 중 오류: {e}")
            return "죄송합니다. 응답을 생성할 수 없습니다."


class DrugQuestionAnswerer:
    """약물 관련 질문 응답 클래스"""
    
    def __init__(self):
        self.classifier = QuestionClassifier()
    
    def answer_drug_question(self, pill_info: Dict[str, Any], question: str) -> str:
        """약물 정보 기반 질문 응답"""
        question_type = self.classifier.classify_drug_question(question)
        
        answer_map = {
            "efficacy": pill_info.get("효능효과", "정보 없음"),
            "dosage": pill_info.get("용법용량", "정보 없음"),
            "precaution": self._get_precaution_info(pill_info),
            "ingredient": pill_info.get("성분정보", "정보 없음"),
            "storage": pill_info.get("저장방법", "정보 없음"),
            "expiry": pill_info.get("사용기간", "정보 없음"),
            "adult_use": self._get_adult_use_info(pill_info),
            "medical_condition": self._get_medical_condition_info(pill_info)
        }
        
        return answer_map.get(question_type, "죄송합니다. 해당 질문은 이해하지 못했습니다.")
    
    def _get_precaution_info(self, pill_info: Dict[str, Any]) -> str:
        """주의사항 정보 조합"""
        atpn = str(pill_info.get("사용상 주의사항", "") or "")
        caution = str(pill_info.get("사용시주의사항", "") or "")
        return atpn + "\n\n" + caution if atpn.strip() or caution.strip() else "정보 없음"
    
    def _get_adult_use_info(self, pill_info: Dict[str, Any]) -> str:
        """성인 복용 관련 정보"""
        caution = str(pill_info.get("사용상 주의사항", "") or "") + " " + str(pill_info.get("사용시주의사항", "") or "")
        if "소아" in caution or "어린이" in caution:
            return "성인 복용 관련 주의사항:\n" + caution
        return "성인 복용에 특별한 제한사항은 언급되어 있지 않습니다."
    
    def _get_medical_condition_info(self, pill_info: Dict[str, Any]) -> str:
        """기저 질환 관련 정보"""
        caution = str(pill_info.get("사용상 주의사항", "") or "")
        return "기저 질환 관련 주의사항:\n" + (caution if caution.strip() else "정보 없음")