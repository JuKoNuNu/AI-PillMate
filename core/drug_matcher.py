"""
약물 매칭 및 검색 모듈 - 완전버전
"""
import pandas as pd
import difflib
import os
from typing import List, Dict, Any, Optional
from utils.text_processor import TextProcessor


class DrugMatcher:
    """약물 매칭 클래스"""
    
    def __init__(self, drug_data: pd.DataFrame):
        self.drug_data = drug_data
        self.text_processor = TextProcessor()
        self._prepare_data()
    
    def _prepare_data(self):
        """데이터 전처리"""
        # 필요한 컬럼들의 정리된 버전 생성
        for col in ['제품명', '업체명', '식별표기']:
            if col in self.drug_data.columns:
                self.drug_data[f'{col}_clean'] = self.drug_data[col].apply(
                    self.text_processor.clean_string
                )
    
    def find_candidates_by_identifiers(self, identifiers: List[str]) -> pd.DataFrame:
        """식별표기로 약물 후보 찾기"""
        if not identifiers:
            return pd.DataFrame()
        
        combined_identifiers = list(set([
            self.text_processor.clean_string(text) 
            for text in identifiers if text.strip()
        ]))
        
        candidates = pd.DataFrame()
        
        for ident in combined_identifiers:
            # 정확한 매치 시도
            exact_match = self.drug_data[
                self.drug_data['식별표기_clean'] == ident
            ]
            
            if not exact_match.empty:
                candidates = pd.concat([candidates, exact_match])
            else:
                # 유사한 매치 시도
                similar_match = self.drug_data[
                    self.drug_data['식별표기_clean'].apply(
                        lambda x: any(difflib.get_close_matches(ident, [x], cutoff=0.6))
                    )
                ]
                candidates = pd.concat([candidates, similar_match])
        
        return candidates.drop_duplicates().reset_index(drop=True)
    
    def find_related_drugs(self, query: str, top_n: int = 5) -> pd.DataFrame:
        """효능효과나 제품명으로 관련 약물 찾기 - 원본 함수"""
        self.drug_data.columns = self.drug_data.columns.str.strip()
        query = query.strip().lower()

        effect_mask = self.drug_data['효능효과'].astype(str).str.lower().str.contains(query, na=False)
        name_mask = self.drug_data['제품명'].astype(str).str.lower().str.contains(query, na=False)
        mask = effect_mask | name_mask

        matched = self.drug_data[mask]

        if matched.empty:
            return pd.DataFrame()

        matched = matched[matched.apply(self._has_valid_image, axis=1)]

        if matched.empty:
            return pd.DataFrame()

        if '사용시주의사항' not in matched.columns:
            matched['사용시주의사항'] = ""

        columns_needed = ['제품명', '효능효과', '구분', '저장_이미지_파일명', '사용시주의사항']
        available_columns = [col for col in columns_needed if col in matched.columns]
        
        return matched[available_columns].drop_duplicates().head(top_n)
    
    def _has_valid_image(self, row) -> bool:
        """유효한 이미지가 있는지 확인"""
        img = row.get('저장_이미지_파일명', '')
        return (isinstance(img, str) and 
                img.strip() and 
                os.path.isfile(os.path.join("images", img)))


class DrugInteractionChecker:
    """약물 상호작용 검사 클래스"""
    
    def __init__(self, pregnancy_data: pd.DataFrame):
        self.pregnancy_data = pregnancy_data
        self._prepare_pregnancy_data()
    
    def _prepare_pregnancy_data(self):
        """임산부 데이터 전처리"""
        if not self.pregnancy_data.empty:
            self.pregnancy_data['제품명_clean'] = (
                self.pregnancy_data['제품명'].str.strip().str.replace(" ", "")
            )
    
    def check_pregnancy_warnings(self, pill_list: List[Dict[str, Any]]) -> str:
        """임산부 금기 약물 확인"""
        warnings = []
        
        for pill in pill_list:
            product_name = pill.get("제품명", "").strip().replace(" ", "")
            match = self.pregnancy_data[
                self.pregnancy_data["제품명_clean"] == product_name
            ]
            
            if not match.empty:
                detail = match.iloc[0]
                grade = detail.get("금기등급", "N/A")
                info = detail.get("상세정보", "상세정보 없음")
                
                warnings.append(
                    f"🚨 **{pill['제품명']}** 은(는) 임산부 금기 약물입니다.\n"
                    f"- 등급: **{grade}등급**\n"
                    f"- 사유: {info[:150]}..."
                )
            else:
                warnings.append(
                    f"✅ **{pill['제품명']}** 은(는) 현재 임산부 금기 약물 목록에 없습니다.\n\n"
                    f"💬 복용하셔도 됩니다. 다만 다른 약물과 함께 복용 중이신가요? 상호작용 가능성을 꼭 확인해주세요.\n\n"
                    f"📌 본 정보는 **식약처 공식 데이터**를 기반으로 안내되며, 자세한 사항은 **전문가와 상담** 바랍니다."
                )
        
        return "\n\n---\n\n".join(warnings)
    
    def check_drug_interaction_summary(self, pill_list: List[Dict]) -> str:
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
    
    def check_ingredient_overlap_and_dosage(self, pill_list: List[Dict], quantities: List[int]) -> str:
        """성분 중복 및 권장량 초과 확인 (원본 함수)"""
        ingredient_total = {}
        warnings = []

        for pill, qty in zip(pill_list, quantities):
            raw = pill.get("성분정보", "")
            extracted = TextProcessor.extract_ingredients(raw)
            for name, amount, unit in extracted:
                try:
                    mg = float(amount)
                    if unit in ["g"]:
                        mg *= 1000
                    total_mg = mg * qty
                    ingredient_total[name] = ingredient_total.get(name, 0) + total_mg
                except:
                    continue

        # 하루 권장량 임의 기준 예시 
        recommended_limits = {
            "아세트아미노펜": 4000,
            "이부프로펜": 2400,
        }

        for name, total in ingredient_total.items():
            if name in recommended_limits and total > recommended_limits[name]:
                warnings.append(f"- ! **{name}**: 총 {total}mg 복용은 1일 권장량 {recommended_limits[name]}mg 초과입니다.")

        # 중복 성분 확인 - 원본 로직 보존
        all_ingredients = sum([TextProcessor.extract_ingredients(p.get("성분정보", "")) for p in pill_list], [])
        duplicates = [name for name, count in pd.Series([n for (n, _, _) in all_ingredients]).value_counts().items() if count > 1]
        
        if duplicates:
            warnings.append("- ! **중복 성분**: " + ", ".join(duplicates) + " 복용 주의 필요")

        if not warnings:
            return "성분 중복 및 권장량 초과 없이 복용 가능합니다."
        return "### ! 성분 및 용량 주의사항\n" + "\n".join(warnings)


class DrugQuestionAnswerer:
    """약물 관련 질문 응답 클래스 - 원본 함수 포함"""
    
    def chat_with_pill_info(self, pill_info: Dict[str, Any], question: str, selected_pills=None) -> str:
        """원본 챗봇 질의 함수"""
        from utils.text_processor import QuestionClassifier
        
        question = question.lower().strip()
        classifier = QuestionClassifier()

        # 병용 질문 감지: selected_pills가 있고 병용 질문이면 바로 병용 분석 실행
        if selected_pills and classifier.is_combination_question(question):
            checker = DrugInteractionChecker(pd.DataFrame())
            return checker.check_drug_interaction_summary(selected_pills)

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