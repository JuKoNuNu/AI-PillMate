#pip install streamlit pandas pillow pytesseract requests
# 실행 : streamlit run app.py


import os
import re
import requests
import difflib
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import easyocr
import streamlit as st
from dotenv import load_dotenv
import openai
from openai import OpenAI
from llama_index.core import (
    VectorStoreIndex,
    Document,
    StorageContext,
    load_index_from_storage,
    Settings
)
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever


load_dotenv()
# print(os.environ['OPENAI_API_KEY'])
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
client = openai.OpenAI(api_key=OPENAI_API_KEY)  # 수정 예정
openai_key = os.getenv("OPENAI_API_KEY") # 수정 예정
google_key = os.getenv("GOOGLE_API_KEY")
google_cse = os.getenv("GOOGLE_CSE_ID")


st.set_page_config(page_title="AI 기반 약 인식 시스템", layout="wide")
reader = easyocr.Reader(['en', 'ko'])

def easyocr_extract_text(image: Image.Image):
    img_array = np.array(image)
    result = reader.readtext(img_array, detail=0)
    return " ".join(result)

def crop_pill_area(image):
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
        text = reader.readtext(roi, detail=0)
        if any(text):
            cropped = img_cv[y:y + h, x:x + w]
            return Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    return image

def clean_str(s):
    return ''.join(filter(str.isalnum, str(s))).upper() if isinstance(s, str) else ''

def extract_ingredients(text):
    return re.findall(r'([가-힣A-Za-z]+)\s*([\d\.]+)\s*(mg|g|㎎|밀리그램)', text)

# 질의 
def chat_with_pill_info(pill_info, question):
    question = question.lower()
    other_drugs = ["타이레놀", "이부프로펜", "부루펜", "아스피린"]
    if any(keyword in question for keyword in ["효능", "기능", "어떤 질병", "무슨 병"]):
        return pill_info.get("효능효과", "정보 없음")
    elif any(keyword in question for keyword in ["복용", "언제", "얼마나", "용법", "용량", "몇 번"]):
        return pill_info.get("용법용량", "정보 없음")
    elif any(keyword in question for keyword in ["주의", "주의사항", "조심", "금기", "경고", "지켜야"]):
        atpn = str(pill_info.get("사용상 주의사항", "") or "")
        caution = str(pill_info.get("사용시주의사항", "") or "")
        return atpn + "\n\n" + caution if atpn.strip() or caution.strip() else "정보 없음"
    elif any(keyword in question for keyword in ["성분", "무엇이", "들어있"]):
        return pill_info.get("성분정보", "정보 없음")
    elif any(keyword in question for keyword in ["보관", "저장"]):
        return pill_info.get("저장방법", "정보 없음")
    elif any(keyword in question for keyword in ["사용기한", "언제까지", "유통기한", "기한"]):
        return pill_info.get("사용기간", "정보 없음")
    elif any(keyword in question for keyword in ["성인", "어른", "성인이 먹어도", "성인이 복용"]):
        caution = str(pill_info.get("사용상 주의사항", "") or "") + " " + str(pill_info.get("사용시주의사항", "") or "")
        if "소아" in caution or "어린이" in caution:
            return "성인 복용 관련 주의사항:\n" + caution
        return "성인 복용에 특별한 제한사항은 언급되어 있지 않습니다."
    elif any(name in question for name in other_drugs) and any(keyword in question for keyword in ["같이", "병용", "동시에", "함께", "복용"]):
        inter = str(pill_info.get("상호작용", "") or "") + " " + str(pill_info.get("사용상 주의사항", "") or "")
        if inter.strip():
            return f"병용 관련 주의사항:\n" + inter
        return f"병용 복용에 대한 정보는 제공되지 않았습니다."
    elif any(keyword in question for keyword in ["임신", "임산부","임부"]):
        atpn = str(pill_info.get("사용상 주의사항", "") or "")
        if "임부" in atpn or "임신" in atpn:
            return "임신 중 복용 주의사항:\n" + atpn
        return "임신 중 복용에 대한 특별한 주의사항은 제공되지 않았습니다."
    elif any(keyword in question for keyword in ["질병", "간", "신장", "고혈압", "당뇨"]):
        caution = str(pill_info.get("사용상 주의사항", "") or "")
        return "기저 질환 관련 주의사항:\n" + (caution if caution.strip() else "정보 없음")
    else:
        return "죄송합니다. 해당 질문은 이해하지 못했습니다."

def check_drug_interaction_summary(pill_list):
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

def check_ingredient_overlap_and_dosage(pill_list, quantities):
    ingredient_total = {}
    warnings = []

    for pill, qty in zip(pill_list, quantities):
        raw = pill.get("성분정보", "")
        extracted = extract_ingredients(raw)
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

    duplicates = [name for name, count in pd.Series([n for (n, _, _) in sum([extract_ingredients(p.get("성분정보", "")) for p in pill_list], [])]).value_counts().items() if count > 1]
    if duplicates:
        warnings.append("- ! **중복 성분**: " + ", ".join(duplicates) + " 복용 주의 필요")

    if not warnings:
        return "성분 중복 및 권장량 초과 없이 복용 가능합니다."
    return "### ! 성분 및 용량 주의사항\n" + "\n".join(warnings)

def is_combination_question(question: str) -> bool:
    comb_keywords = ["병용", "함께", "같이", "동시", "복용", "병용금기", "병용주의", "병용투여", "복합"]
    q = question.lower()
    return any(k in q for k in comb_keywords)

# 엑셀 데이터 로딩
@st.cache_data
def load_excel_data():
    return pd.read_excel("txta.xlsx")

# 문서 로딩 (RAG용)
def load_docs_from_excel(filepath):
    df_local = pd.read_excel(filepath)
    documents = []
    for _, row in df_local.iterrows():
        content = f"""
        제품명: {row.get('제품명')}
        효능효과: {row.get('효능효과')}
        구분: {row.get('구분')}
        사용시주의사항: {row.get('사용시주의사항', '')}
        저장_이미지_파일명: {row.get('저장_이미지_파일명', '')}
        """
        documents.append(Document(text=content.strip()))
    return documents

# LlamaIndex 쿼리 엔진 설정

@st.cache_resource(show_spinner=True)
def get_query_engine():
    persist_dir = "./index_store"
    
    if not os.path.exists(persist_dir):
        docs = load_docs_from_excel("txta.xlsx")
        index = VectorStoreIndex.from_documents(docs)
        index.storage_context.persist(persist_dir=persist_dir)

    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    index = load_index_from_storage(storage_context)

    # LLM 설정
    llm = LlamaOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        system_prompt=(
            "당신은 한국어로만 대답해야합니다.\n"
            "당신은 약사로서 약에 대한 정보를 사용자의 질문에 맞게 친절하고 자세하게 설명해주어야 합니다."
            "예시"
            "감기에 걸린거 같아 -> 당신은 감기에 걸렸습니다. 감기의 증상은 열, 기침, 콧물 등이 있으며 이 약 '제품명'은 이러한 증상에 효과적입니다. 사용방법은 XXX, 부작용이나 주의사항은 XXX입니다."
            "진짜 약사가 설명하듯이 설명해야하며 최소 2줄은 넘어야 합니다."
            "제품명 : XXX\n"
            "구분 : XXX\n"
            "효능효과 : XXX\n\n"
            "그 다음 줄에 다음 지침을 따릅니다:\n"
            "\n"

        )
    )
    Settings.llm = llm

    # 구성
    retriever = VectorIndexRetriever(index=index, similarity_top_k=2)
    response_synthesizer = get_response_synthesizer(response_mode="tree_summarize")

    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer
    )

    return query_engine

#기본적인 대화
def is_small_talk(text):
    small_talk_keywords = ["안녕", "고마워", "잘자", "뭐해", "하이", "반가워", "좋은 하루", "ㅎㅇ", "hello", "hi","고맙습니다","반가워","꺼져","음"]
    return any(keyword in text.lower() for keyword in small_talk_keywords)

# 구글 이미지 검색 (이미지 URL 리스트 반환)
def google_image_search(api_key, cse_id, query, num=1):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": api_key,
        "cx": cse_id,
        "q": query,
        "searchType": "image",
        "num": num,
    }
    response = requests.get(url, params=params)
    results = response.json()
    return [item["link"] for item in results.get("items", [])]

# Streamlit에 이미지 표시 함수
def show_images(image_urls):
    for url in image_urls:
        st.image(url, width=300)

# GPT로 핵심 키워드 추출
def extract_keyword(question):
    messages = [
            {"role": "system", "content": (
            "당신은 한국어 질문에서 핵심 키워드를 정확히 한 단어로 추출하는 AI 약사입니다. "
            "사용자의 질문에서 가장 중요한 증상 또는 약과 관련된 단어만 뽑아야 합니다. "
            "예시:\n"
            "'감기 걸린 것 같아' → '감기'\n"
            "'두통이 심해요' → '두통'\n"
            "'어지럽고 기운이 없어요' → '어지럼증'\n"
            "'목이 따끔거리고 열이 나요' → '목감기'\n"
            "정확히 한 단어로만 출력하세요."
        )},
            {"role": "user", "content": f"질문: {question}\n핵심 키워드 한 단어만 출력해줘"}
    ]
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.2,
        max_tokens=10,
    )
    return response.choices[0].message.content.strip()

# 약 검색 함수 (효능효과 또는 제품명 포함 여부 확인)
def find_related_drugs(excel_df, user_query, top_n=5):
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

def ocr_page():
    st.title("약 이미지 인식 기반 정보 제공 시스템")

    marge_all = pd.read_excel("final_data.xlsx")
    for col in ['제품명', '업체명', '식별표기']:
        marge_all[col] = marge_all[col].astype(str)
        marge_all[f'{col}_clean'] = marge_all[col].apply(clean_str)

    uploaded_files = st.file_uploader("궁금하신 약 이미지들 업로드 해주세요", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        st.header("업로드한 약 식별표기 확인")
        edited_texts = []

        for i, uploaded_file in enumerate(uploaded_files):
            image = Image.open(uploaded_file).convert("RGB")
            cropped = crop_pill_area(image)
            st.image(cropped, caption=f"이미지 {i+1} 크롭 약 이미지", width=200)

            ocr_text = easyocr_extract_text(cropped).upper()
            edited = st.text_input(f"약 {i+1} 식별표기 (이미지 인식이 잘못 되었다면 수정해주세요)", value=ocr_text, key=f"ocr_edit_{i}")
            edited_texts.append(edited)

        combined_identifiers = list(set([clean_str(text) for text in edited_texts if text.strip()]))

        st.header("식별표기 확인 후 약을 선택하고 개수를 지정하세요")

        candidates = pd.DataFrame()
        for ident in combined_identifiers:
            temp = marge_all[marge_all['식별표기_clean'] == ident]
            if temp.empty:
                temp = marge_all[marge_all['식별표기_clean'].apply(
                    lambda x: any(difflib.get_close_matches(ident, [x], cutoff=0.6))
                )]
            candidates = pd.concat([candidates, temp])

        candidates = candidates.drop_duplicates().reset_index(drop=True)

        if candidates.empty:
            st.warning("조건에 맞는 약 후보가 없습니다.")
            return

        if 'basket' not in st.session_state:
            st.session_state['basket'] = {}

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

        basket_items = [displayed_candidates[key] for key in st.session_state['basket'].keys()]
        quantities = [st.session_state['basket'][key]['quantity'] for key in st.session_state['basket'].keys()]

        if basket_items:
            st.header("선택한 약과 수량을 확인하고 궁금한 점을 입력하세요")
            question = st.text_input("ex) 보관방법이 어떻게 되나요?, 이 약들을 같이 먹어도 괜찮나요?")

            if question:
                if is_combination_question(question):
                    # 병용 질문일 때
                    st.markdown(check_drug_interaction_summary(basket_items))
                    st.markdown(check_ingredient_overlap_and_dosage(basket_items, quantities))
                else:
                    # 개별 약 질문일 때
                    for i, pill_info in enumerate(basket_items):
                        st.subheader(f"약 {i+1}: {pill_info['제품명']}")
                        answer = chat_with_pill_info(pill_info, question)
                        st.write(answer)

                    st.markdown(check_drug_interaction_summary(basket_items))
                    st.markdown(check_ingredient_overlap_and_dosage(basket_items, quantities))

        else:
            st.info("선택한 약이 없습니다.")

# 메인 함수
def chatbot_page():
    st.title("AI 기반 다제약물(중복 복용) 예방 챗봇")
    st.markdown("**약물 관련 궁금한 점이나 증상을 입력하면 추천 약을 보여드립니다.**")

    # 사용자 상태 초기화
    if 'forbid' not in st.session_state:
        st.session_state.forbid = '해당 사항 없음'
    
    # 선택된 약품 목록 초기화
    if 'selected_drugs' not in st.session_state:
        st.session_state.selected_drugs = []
        
    # 요청 상태 초기화 (NEW: 요청 버튼 상태 관리)
    if 'request_submitted' not in st.session_state:
        st.session_state.request_submitted = False

    st.subheader("⚠️ 사용자 상태 선택")
    forbid_options = ['임산부', '노인', '중복의약품 문의', '해당 사항 없음']    # 버튼 작업 완료
    cols = st.columns(len(forbid_options))
    for i, option in enumerate(forbid_options):
        if cols[i].button(option, key=f"btn_{i}"):
            st.session_state.forbid = option
            # 상태가 변경되면 선택된 약품 목록 초기화
            st.session_state.selected_drugs = []
            # 요청 상태 초기화 (NEW)
            st.session_state.request_submitted = False
            st.success(f"선택된 사용자 상태: {option}")
    
    # 데이터
    marge_all = pd.read_excel("final_data.xlsx")
    main_df, preg_df, old_df, dup_df =  marge_all

    # [UPDATED] 중복의약품 문의 전용 입력 구성 - 반응형 개선 및 중복 ID 해결
    placeholder = st.empty()
    
    # 중복의약품 문의 전용 입력 및 처리
    if st.session_state.forbid == '중복의약품 문의':
        # 중복의약품 문의 전용 입력 필드
        dup_question = st.text_input("📝 질문 또는 증상을 입력하세요 (예: 중복복용 등)", key="dup_question_input")
        
        with placeholder.container():
            st.markdown("### 💊 현재 드시고 계신 의약품 관련 정보 입력")

            with st.expander("중복 복용 주의 의약품 카테고리 확인하기", expanded=True):
                dup_categories = [
                    '선택하세요', '혈압강하작용의약품', '당뇨병용제', '지질저하제', '소화성궤양용제',
                    '해열진통소염제', '정신신경용제', '호흡기관용약', '마약류 아편유사제', '최면진정제'
                ]
                selected_category = st.selectbox("📂 카테고리 선택", dup_categories, key="dup_category_select")

                if selected_category != '선택하세요':
                    related_dup_drugs = dup_df[dup_df['효능군'] == selected_category]
                    
                    if not related_dup_drugs.empty:
                        st.markdown(f"#### 📋 '{selected_category}' 관련 의약품 목록:")
                        
                        # 최대 10개만 표시 (토큰 제한 방지)
                        display_count = min(10, len(related_dup_drugs))
                        
                        # 중복 키 문제 해결: 인덱스를 추가하여 고유한 키 생성
                        for i in range(display_count):
                            row = related_dup_drugs.iloc[i]
                            product_name = row['제품명']
                            
                            # 고유한 키 생성 방법 - 인덱스 값을 함께 사용
                            unique_key = f"drug_{i}_{product_name}"
                            
                            # 체크박스로 약품 선택 기능 구현 (고유한 키 사용)
                            if st.checkbox(f"{product_name}", key=unique_key):
                                # 선택된 약품 목록에 추가 (중복 방지)
                                if product_name not in st.session_state.selected_drugs:
                                    st.session_state.selected_drugs.append(product_name)
                            elif product_name in st.session_state.selected_drugs:
                                # 체크 해제 시 목록에서 제거
                                st.session_state.selected_drugs.remove(product_name)
                                
                        # 더 많은 약품이 있으면 알림
                        if len(related_dup_drugs) > 10:
                            st.info(f"표시된 약품 외에 {len(related_dup_drugs) - 10}개 더 있습니다. 검색 기능을 사용해보세요.")
                    else:
                        st.info("선택한 카테고리에 해당하는 의약품 정보가 없습니다.")
            
            # 선택된 약품 목록 표시
            if st.session_state.selected_drugs:
                st.markdown("### 🔍 선택된 약품 목록")
                for selected_drug in st.session_state.selected_drugs:
                    st.markdown(f"- {selected_drug}")
        

        # GPT 응답 & 약물 정보 제공공 (중복의약품 문의) - 요청 버튼 추가 (NEW)
        if dup_question and st.session_state.selected_drugs:
            # 요청 버튼 추가 (NEW)
            request_button = st.button("📤 요청하기", key="submit_dup_request")
            
            # 요청 버튼을 누르면 요청 상태를 True로 설정 (NEW)
            if request_button:
                st.session_state.request_submitted = True
                st.success("요청이 제출되었습니다.")

                # --- 현재 작업중 ------------------------------------------------------------------------------------------------------------------------------

                if dup_question and st.session_state.request_submitted:
                    with st.spinner("관련 약품을 찾는 중입니다..."):
                        drug_df = find_related_drugs(main_df, selected_drug)

                    if drug_df.empty:
                        st.error("❌ 해당 키워드와 관련된 약품을 찾을 수 없습니다.")
                        
                    else:
                        # 결과 개수 제한 (최대 5개)
                        display_count = min(5, len(drug_df))
                        limited_drug_df = drug_df.head(display_count)
                        
                        drug_summary_text = "🔎 관련 약품 목록:\n"
                        for _, row in limited_drug_df.iterrows():
                            name = row['제품명']
                            image_filename = row['저장_이미지_파일명']
                            image_path = os.path.join("images", image_filename)

                            if os.path.exists(image_path):
                                st.image(image_path, caption=name, width=300)
                            else:
                                st.warning(f"이미지 없음: {image_filename}")

                            st.markdown(f"**💊 {name}** ({row['구분']})")
                            st.markdown(f"**효능:** {row['효능효과']}")

                            precautions = row.get("주의사항_병합", "")
                            if pd.notna(precautions) and str(precautions).strip() != "":
                                with st.expander("📌 사용 시 주의사항 보기"):
                                    st.markdown(str(precautions))
                            else:
                                st.markdown("ℹ️ 사용 시 주의사항 정보 없음")


                # --- 현재 작업중 -------------------------------------------------------------------------------------------------------------------------------
            
            # 요청 버튼이 눌려진 상태일 때만 API 호출 실행 (NEW)
            if st.session_state.request_submitted:
                # 선택된 약품만 포함하여 요약 생성 (토큰 제한 문제 해결)
                drug_dup_summary_text = "🔎 관련 약품 목록:\n"
                for drug in st.session_state.selected_drugs:
                    drug_dup_summary_text += f"- {drug}\n"
                
                messages = [
                    {"role": "system", "content": "당신은 약물 복용 정보와 병용 금기 등에 대해 답변하는 약사 AI입니다."},
                    {"role": "user", "content": f"아래 약 정보 참고해서 답변해주세요:\n{drug_dup_summary_text}\n\n질문: {dup_question}"}
                ]
                
                # 디버깅용 메시지 출력 (선택사항)
                print(messages)
                
                try:
                    response = client.chat.completions.create(
                        model=os.environ['OPENAI_API_MODEL'],
                        messages=messages,
                        temperature=0.5,
                        max_tokens=500,
                    )
                    answer = response.choices[0].message.content
                    st.markdown("### 💡 GPT 답변:")
                    st.markdown(answer)
                    
                except Exception as e:
                    st.error(f"GPT 응답 생성 중 오류 발생: {e}")
                    st.info("💡 팁: 선택한 약품이 너무 많으면 토큰 제한에 걸릴 수 있습니다. 필요한 약품만 선택해주세요.")
        elif dup_question and not st.session_state.selected_drugs:
            st.warning("⚠️ 질문하기 전에 약품을 하나 이상 선택해주세요.")
    else:
        # 일반 질문 입력 필드 (중복의약품 문의가 아닐 때만 표시)
        user_question = st.text_input("📝 질문 또는 증상을 입력하세요 (예: 감기, 비타민 등)", key="general_question_input")
        
        # 요청 버튼 추가 (NEW)
        if user_question:
            request_button = st.button("📤 요청하기", key="submit_general_request")
            
            # 요청 버튼을 누르면 요청 상태를 True로 설정 (NEW)
            if request_button:
                st.session_state.request_submitted = True
                st.success("요청이 제출되었습니다.")
        
        # 일반 질문에 대한 처리 - 요청 버튼이 눌려진 상태일 때만 실행 (NEW)
        if user_question and st.session_state.request_submitted:
            with st.spinner("관련 약품을 찾는 중입니다..."):
                drug_df = find_related_drugs(main_df, user_question)

            if drug_df.empty:
                st.error("❌ 해당 키워드와 관련된 약품을 찾을 수 없습니다.")
                
            else:
                # 결과 개수 제한 (최대 5개)
                display_count = min(5, len(drug_df))
                limited_drug_df = drug_df.head(display_count)
                
                drug_summary_text = "🔎 관련 약품 목록:\n"
                for _, row in limited_drug_df.iterrows():
                    name = row['제품명']
                    image_filename = row['저장_이미지_파일명']
                    image_path = os.path.join("images", image_filename)

                    if os.path.exists(image_path):
                        st.image(image_path, caption=name, width=300)
                    else:
                        st.warning(f"이미지 없음: {image_filename}")

                    st.markdown(f"**💊 {name}** ({row['구분']})")
                    st.markdown(f"**효능:** {row['효능효과']}")

                    precautions = row.get("주의사항_병합", "")
                    if pd.notna(precautions) and str(precautions).strip() != "":
                        with st.expander("📌 사용 시 주의사항 보기"):
                            st.markdown(str(precautions))
                    else:
                        st.markdown("ℹ️ 사용 시 주의사항 정보 없음")

                    # 사용자 상태에 따라 추가 정보 출력
                    if st.session_state.forbid == '임산부':
                        preg_info = preg_df[preg_df['제품명'] == name]
                        if not preg_info.empty:
                            st.error("🚨 임산부 금기 약물입니다!")
                            st.markdown(f"**금기등급:** {preg_info.iloc[0]['금기등급']}")
                            st.markdown(f"**상세정보:** {preg_info.iloc[0]['상세정보']}")

                    elif st.session_state.forbid == '노인':
                        old_info = old_df[old_df['제품명'] == name]
                        if not old_info.empty:
                            st.warning("⚠️ 노인 주의 약물입니다.")
                            st.markdown(f"**약품상세정보:** {old_info.iloc[0]['약품상세정보']}")

                    st.markdown("---")
                    drug_summary_text += f"- {name}\n"
                
                # 더 많은 결과가 있음을 알림
                if len(drug_df) > display_count:
                    st.info(f"⚠️ 검색 결과가 많아 상위 {display_count}개만 표시합니다. (총 {len(drug_df)}개 검색됨)")

                # GPT 응답 (일반 질문)
                messages = [
                    {"role": "system", "content": "당신은 약물 복용 정보와 병용 금기 등에 대해 답변하는 약사 AI입니다."},
                    {"role": "user", "content": f"아래 약 정보 참고해서 답변해주세요:\n{drug_summary_text}\n\n질문: {user_question}"}
                ]
                
                # 디버깅용 메시지 출력 (선택사항)
                print(messages)

                try:
                    response = client.chat.completions.create(
                        model=os.environ['OPENAI_API_MODEL'],
                        messages=messages,
                        temperature=0.5,
                        max_tokens=500,
                    )
                    answer = response.choices[0].message.content
                    st.markdown("### 💡 GPT 답변:")
                    st.markdown(answer)
                except Exception as e:
                    st.error(f"GPT 응답 생성 중 오류 발생: {e}")

# 챗봇 페이지 구현
def chatbot2_page():
    st.title("💊 AI 기반 약품 추천 챗봇")
    st.markdown("사용자 질문에 따라 약 정보를 추천합니다.")
    user_question = st.text_input("❓ 증상이나 궁금한 약을 입력하세요\n 예시) 내가 지금 몸이 열이 나고 콧물이 나서 코가 막혀 어떤 약이 좋을까? ")
    df = load_excel_data()

    if user_question:
        if is_small_talk(user_question):
            st.markdown("🤖 **기본 대화 응답입니다.**")
            base_response = f"안녕하세요! 😊 무엇을 도와드릴까요?"
            st.success("✅ AI 응답 완료")
            st.markdown(base_response)

        else:
            st.markdown("🤖 **LlamaIndex 기반 RAG로 답변 중...**")
            query_engine = get_query_engine()
            response = query_engine.query(user_question)
            st.success("✅ AI 응답 완료")

            full_response = response.response

            # 정규식으로 정보 추출
            product_match = re.search(r"제품명\s*[:：]\s*(.+)", full_response)
            class_match = re.search(r"구분\s*[:：]\s*(.+)", full_response)
            effect_match = re.search(r"효능효과\s*[:：]\s*(.+)", full_response)

            product_name = product_match.group(1).strip() if product_match else "알 수 없음"
            product_class = class_match.group(1).strip() if class_match else ""
            effect_text = effect_match.group(1).strip() if effect_match else ""

            answer_text = full_response
            for pattern in [r"제품명\s*[:：].+", r"구분\s*[:：].+", r"효능효과\s*[:：].+"]:
                answer_text = re.sub(pattern, "", answer_text).strip()

            if product_name.lower() != "알 수 없음":
                if product_name.lower() != "xxx":
                    st.markdown(f"### 💊 {product_name} ({product_class})")
                    st.markdown(f"**AI 답변:** {answer_text}")

                    # 🔹 구글 이미지 출력
                    google_img_url = google_image_search(google_key, google_cse, product_name, num=1)
                    if google_img_url:
                        st.image(google_img_url[0], caption=f"{product_name} (검색 이미지)", width=300)
                    else:
                        st.info("🔍 구글 이미지 검색 결과 없음")

                    # 🔹 추가 설명 출력
                    st.markdown("추가적인 정보입니다.")
                    st.markdown(f"💊 {product_name} ({product_class})")
                    st.markdown(f"**효능효과:** {effect_text}")

                    # 🔹 주의사항 출력 (엑셀에서 해당 row 검색)
                    excel_row = df[df["제품명"] == product_name]
                    if not excel_row.empty:
                        warning = excel_row.iloc[0].get("사용시주의사항", "")
                        if isinstance(warning, str) and warning.strip():
                            with st.expander("📌 사용시 주의사항 보기"):
                                st.markdown(warning)
                        else:
                            st.info("⚠️ 주의사항 정보가 없습니다.")
                else:
                    st.warning("❗ 제품명을 찾지 못했습니다.")


def main():
    st.sidebar.title("서비스 선택")
    page = st.sidebar.radio("메뉴", ["알약 이미지 검색", "약 정보 검색 챗봇"])

    if page == "알약 이미지 검색":
        ocr_page()
        chatbot_page()
    elif page == "약 정보 검색 챗봇":
        chatbot2_page()

# 실행
if __name__ == "__main__":
    main()
 # type: ignore