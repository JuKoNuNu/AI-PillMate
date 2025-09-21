"""
텍스트 처리 유틸리티 모듈
"""
import re
from typing import List, Tuple, Dict, Any


class TextProcessor:
    """텍스트 처리 클래스"""
    
    @staticmethod
    def clean_string(s: str) -> str:
        """문자열 정리 (알파벳과 숫자만 남기고 대문자 변환)"""
        return ''.join(filter(str.isalnum, str(s))).upper() if isinstance(s, str) else ''
    
    @staticmethod
    def extract_ingredients(text: str) -> List[Tuple[str, str, str]]:
        """성분 정보 추출"""
        return re.findall(r'([가-힣A-Za-z]+)\s*([\d\.]+)\s*(mg|g|㎎|밀리그램)', text)


class QuestionClassifier:
    """질문 분류 클래스"""
    
    @staticmethod
    def is_small_talk(text: str) -> bool:
        """일반적인 인사말인지 판단"""
        small_talk_keywords = [
            "안녕", "고마워", "잘자", "뭐해", "하이", "반가워", 
            "좋은 하루", "ㅎㅇ", "hello", "hi", "고맙습니다", "반가워", "음"
        ]
        return any(keyword in text.lower() for keyword in small_talk_keywords)
    
    @staticmethod
    def is_combination_question(question: str) -> bool:
        """병용 관련 질문인지 판단"""
        comb_keywords = [
            "병용", "함께", "같이", "동시", "복용", "병용금기", 
            "병용주의", "병용투여", "복합"
        ]
        return any(k in question.lower() for k in comb_keywords)
    
    @staticmethod
    def is_pregnancy_question(question: str) -> bool:
        """임신 관련 질문인지 판단"""
        return any(k in question.lower() for k in ["임산부", "임신", "태아"])
    
    @staticmethod
    def is_followup_question(text: str) -> bool:
        """후속 질문인지 판단"""
        followup_keywords = [
            "다른 약", "또 있어", "추천 더 해줘", "비슷한 약", "대체 약", 
            "더 없을까", "비슷한", "또", "말고", "다른거", "다른건?"
        ]
        return any(keyword in text.lower() for keyword in followup_keywords)
    
    @staticmethod
    def classify_drug_question(question: str) -> str:
        """약물 관련 질문 분류"""
        question_lower = question.lower().strip()
        
        # 효능 관련
        if any(keyword in question_lower for keyword in ["효능", "기능", "어떤 질병", "무슨 병"]):
            return "efficacy"
        
        # 복용법 관련
        elif any(keyword in question_lower for keyword in ["복용", "언제", "얼마나", "용법", "용량", "몇 번"]):
            return "dosage"
        
        # 주의사항 관련
        elif any(keyword in question_lower for keyword in ["주의", "주의사항", "조심", "금기", "경고", "지켜야"]):
            return "precaution"
        
        # 성분
        elif any(keyword in question_lower for keyword in ["성분", "무엇이", "들어있"]):
            return "ingredient"
        
        # 보관
        elif any(keyword in question_lower for keyword in ["보관", "저장"]):
            return "storage"
        
        # 사용기한
        elif any(keyword in question_lower for keyword in ["사용기한", "언제까지", "유통기한", "기한"]):
            return "expiry"
        
        # 성인 관련
        elif any(keyword in question_lower for keyword in ["성인", "어른", "성인이 먹어도", "성인이 복용"]):
            return "adult_use"
        
        # 기저 질환 관련
        elif any(keyword in question_lower for keyword in ["질병", "간", "신장", "고혈압", "당뇨"]):
            return "medical_condition"
        
        else:
            return "unknown"
        
def chat_with_pill_info(pill_info: Dict[str, Any], question: str, selected_pills=None) -> str:
    """원본 챗봇 질의 함수"""
    from utils.text_processor import QuestionClassifier
    
    question = question.lower().strip()
    classifier = QuestionClassifier()

    # 병용 질문 감지: selected_pills가 있고 병용 질문이면 바로 병용 분석 실행
    if selected_pills and classifier.is_combination_question(question):
        return check_drug_interaction_summary(selected_pills)

    # 효능 관련
    elif any(keyword in question for keyword in ["효능", "기능", "어떤 질병", "무슨 병"]):
        return pill_info.get("효능효과", "정보 없음")

    # 복용법 관련
    elif any(keyword in question for keyword in ["복용", "언제", "얼마나", "용법", "용량", "몇 번"]):
        return pill_info.get("용법용량", "정보 없음")

    # 주의사항 관련
    elif any(keyword in question for keyword in ["주의", "주의사항", "조심", "금기", "경고", "지켜야"]):
        atpn = str(pill_info.get("사용상 주의사항", "") or "")
        caution = str(pill_info.get("사용시주의사항", "") or "")
        return atpn + "\n\n" + caution if atpn.strip() or caution.strip() else "정보 없음"

    # 성분
    elif any(keyword in question for keyword in ["성분", "무엇이", "들어있"]):
        return pill_info.get("성분정보", "정보 없음")

    # 보관
    elif any(keyword in question for keyword in ["보관", "저장"]):
        return pill_info.get("저장방법", "정보 없음")

    # 사용기한
    elif any(keyword in question for keyword in ["사용기한", "언제까지", "유통기한", "기한"]):
        return pill_info.get("사용기간", "정보 없음")

    # 성인 관련
    elif any(keyword in question for keyword in ["성인", "어른", "성인이 먹어도", "성인이 복용"]):
        caution = str(pill_info.get("사용상 주의사항", "") or "") + " " + str(pill_info.get("사용시주의사항", "") or "")
        if "소아" in caution or "어린이" in caution:
            return "성인 복용 관련 주의사항:\n" + caution
        return "성인 복용에 특별한 제한사항은 언급되어 있지 않습니다."

    # 임신 관련
    elif any(keyword in question for keyword in ["임신", "임산부", "임부"]):
        atpn = str(pill_info.get("사용상 주의사항", "") or "")
        if "임부" in atpn or "임신" in atpn:
            return "임신 중 복용 주의사항:\n" + atpn
        return "임신 중 복용에 대한 특별한 주의사항은 제공되지 않았습니다."

    # 기저 질환 관련
    elif any(keyword in question for keyword in ["질병", "간", "신장", "고혈압", "당뇨"]):
        caution = str(pill_info.get("사용상 주의사항", "") or "")
        return "기저 질환 관련 주의사항:\n" + (caution if caution.strip() else "정보 없음")

    else:
        return "죄송합니다. 해당 질문은 이해하지 못했습니다."


def check_drug_interaction_summary(pill_list: List[Dict]) -> str:
    """병용 가능 - 약 분류 (원본 함수)"""
    warnings = []
    safe_pairs = []

    for i in range(len(pill_list)):
        for j in range(i + 1, len(pill_list)):
            p1 = pill_list[i]
            p2 = pill_list[j]
            p1_name = p1.get("제품명", f"약{i+1}")
            p2_name = p2.get("제품명", f"약{j+1}")
            inter1 = str(p1.get("상호작용", "") or "") + " " + str(p1.get("사용상 주의사항", "") or "")
            inter2 = str(p2.get("상호작용", "") or "") + " " + str(p2.get("사용상 주의사항", "") or "")
            combined = inter1.lower() + " " + inter2.lower()
            if p1_name.lower() in combined or p2_name.lower() in combined or any(
                keyword in combined for keyword in ["병용", "같이", "동시", "복합", "함께"]
            ):
                warnings.append(f"- **{p1_name} ↔ {p2_name}**: 병용 시 주의사항이 있습니다.")
            else:
                safe_pairs.append(f"{p1_name}, {p2_name}")

    result = "### ▶ 선택한 약들 간의 병용 여부 분석\n\n"
    if warnings:
        result += "#### ! 주의가 필요한 조합:\n" + "\n".join(warnings) + "\n\n"
    if safe_pairs:
        result += "#### ! 병용에 특별한 주의사항이 없는 조합:\n" + ", ".join(safe_pairs) + "\n"

    return result