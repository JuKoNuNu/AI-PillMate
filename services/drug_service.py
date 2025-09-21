"""
약물 정보 서비스 모듈
"""
import pandas as pd
from typing import Dict, Any, Optional


class DrugInformationService:
    """약물 정보 서비스 클래스"""
    
    def __init__(self, excel_data: pd.DataFrame):
        self.excel_data = excel_data
    
    def get_usage_instruction(self, product_name: str) -> str:
        """제품명으로 용법용량 정보 조회"""
        if self.excel_data.empty:
            return "없음"
        
        product_name = str(product_name).strip().lower()
        excel_data = self.excel_data.copy()
        excel_data['제품명_정리'] = excel_data['제품명'].astype(str).str.strip().str.lower()
        
        matched = excel_data.loc[
            excel_data['제품명_정리'] == product_name,
            '용법용량'
        ]
        
        if not matched.empty and pd.notna(matched.iloc[0]):
            return matched.iloc[0]
        
        return "없음"
    
    def get_drug_info(self, product_name: str) -> Dict[str, Any]:
        """제품명으로 약물 전체 정보 조회"""
        if self.excel_data.empty:
            return {}
        
        product_name = str(product_name).strip()
        matched = self.excel_data[
            self.excel_data['제품명'].astype(str).str.strip() == product_name
        ]
        
        if matched.empty:
            return {}
        
        return matched.iloc[0].to_dict()
    
    def search_by_effect(self, effect_keyword: str) -> pd.DataFrame:
        """효능효과로 약물 검색"""
        if self.excel_data.empty:
            return pd.DataFrame()
        
        effect_keyword = effect_keyword.lower().strip()
        
        mask = self.excel_data['효능효과'].astype(str).str.lower().str.contains(
            effect_keyword, na=False
        )
        
        return self.excel_data[mask]
    
    def get_safety_info(self, product_name: str) -> Dict[str, str]:
        """안전성 정보 조회"""
        drug_info = self.get_drug_info(product_name)
        
        if not drug_info:
            return {
                "사용상주의사항": "정보 없음",
                "사용시주의사항": "정보 없음",
                "상호작용": "정보 없음"
            }
        
        return {
            "사용상주의사항": drug_info.get("사용상 주의사항", "정보 없음"),
            "사용시주의사항": drug_info.get("사용시주의사항", "정보 없음"),
            "상호작용": drug_info.get("상호작용", "정보 없음")
        }
        
def find_related_drugs(excel_df: pd.DataFrame, user_query: str, top_n: int = 5) -> pd.DataFrame:
    """효능효과 또는 제품명 포함 여부 확인 (원본 함수)"""
    excel_df.columns = excel_df.columns.str.strip()
    query = user_query.strip().lower()

    effect_mask = excel_df['효능효과'].astype(str).str.lower().str.contains(query, na=False)
    name_mask = excel_df['제품명'].astype(str).str.lower().str.contains(query, na=False)
    mask = effect_mask | name_mask

    matched = excel_df[mask]

    if matched.empty:
        return pd.DataFrame()

    def has_valid_image(row):
        img = row.get('저장_이미지_파일명', '')
        return isinstance(img, str) and img.strip() and os.path.isfile(os.path.join("images", img))

    matched = matched[matched.apply(has_valid_image, axis=1)]

    if matched.empty:
        return pd.DataFrame()

    if '사용시주의사항' not in matched.columns:
        matched['사용시주의사항'] = ""

    columns_needed = ['제품명', '효능효과', '구분', '저장_이미지_파일명', '사용시주의사항']
    return matched[columns_needed].drop_duplicates().head(top_n)
