"""
OCR 처리 모듈
"""
import cv2
import numpy as np
import easyocr
from PIL import Image
from typing import List, Optional
from config.settings import Config


class OCRProcessor:
    """OCR 처리 클래스"""
    
    def __init__(self):
        self.config = Config()
        self.reader = easyocr.Reader(self.config.OCR_LANGUAGES)
    
    def extract_text(self, image: Image.Image) -> str:
        """이미지에서 텍스트 추출"""
        try:
            img_array = np.array(image)
            result = self.reader.readtext(img_array, detail=0)
            return " ".join(result)
        except Exception as e:
            print(f"OCR 처리 중 오류: {e}")
            return ""
    
    def crop_pill_area(self, image: Image.Image) -> Image.Image:
        """약 영역 크롭"""
        try:
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (7, 7), 0)
            _, thresh = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY_INV)
            
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return image
            
            for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
                x, y, w, h = cv2.boundingRect(cnt)
                roi = gray[y:y + h, x:x + w]
                text = self.reader.readtext(roi, detail=0)
                
                if any(text):
                    cropped = img_cv[y:y + h, x:x + w]
                    return Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
            
            return image
            
        except Exception as e:
            print(f"이미지 크롭 중 오류: {e}")
            return image
    
    def process_multiple_images(self, uploaded_files: List) -> List[str]:
        """여러 이미지 처리"""
        extracted_texts = []
        
        for uploaded_file in uploaded_files:
            try:
                image = Image.open(uploaded_file).convert("RGB")
                cropped = self.crop_pill_area(image)
                ocr_text = self.extract_text(cropped).upper()
                extracted_texts.append(ocr_text)
            except Exception as e:
                print(f"이미지 처리 중 오류: {e}")
                extracted_texts.append("")
        
        return extracted_texts