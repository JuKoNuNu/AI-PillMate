"""
구글 검색 서비스 모듈
"""
import requests
from typing import List, Optional
from config.settings import Config


class GoogleSearchService:
    """구글 이미지 검색 서비스 클래스"""
    
    def __init__(self):
        self.config = Config()
    
    def search_images(self, query: str, num: int = 1) -> List[str]:
        """구글 이미지 검색"""
        if not self.config.GOOGLE_API_KEY or not self.config.GOOGLE_CSE_ID:
            print("Google API 키 또는 CSE ID가 설정되지 않았습니다.")
            return []
        
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": self.config.GOOGLE_API_KEY,
            "cx": self.config.GOOGLE_CSE_ID,
            "q": query,
            "searchType": "image",
            "num": min(num, 10),  # 최대 10개 제한
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            results = response.json()
            
            return [
                item["link"] 
                for item in results.get("items", [])
                if "link" in item
            ]
        
        except requests.RequestException as e:
            print(f"구글 검색 API 호출 실패: {e}")
            return []
        except Exception as e:
            print(f"구글 검색 중 오류: {e}")
            return []
    
    def search_drug_images(self, drug_name: str) -> List[str]:
        """약물명으로 이미지 검색"""
        if not drug_name.strip():
            return []
        
        # 약물 이미지 검색을 위한 쿼리 최적화
        query = f"{drug_name} 약 의약품"
        return self.search_images(query, num=3)