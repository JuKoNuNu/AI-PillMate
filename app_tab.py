# streamlit run app_tab.py

import streamlit as st
import os
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import easyocr
import difflib
import re
import openai
from openai import OpenAI
from dotenv import load_dotenv
from copy import deepcopy

load_dotenv(dotenv_path=r"C:\\code\\PJT2\\openai_api.env")
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
client = openai.OpenAI(api_key=OPENAI_API_KEY)

old_df = pd.read_excel("pregnant_warning.xlsx")


# -----------------------------------------------------------------------------------------

# 페이지 설정
st.set_page_config(page_title="💡AI 기반 다제약물(중복 복용) 예방 챗봇", layout="wide")

# OCR 리더 로딩
@st.cache_resource
def load_easyocr():
    return easyocr.Reader(['en', 'ko'])

reader = load_easyocr()

# OCR 텍스트 추출 함수
def easyocr_extract_text(image):
    img_array = np.array(image)
    result = reader.readtext(img_array, detail=0)
    return " ".join(result)

# 문자열 정제
def clean_str(s):
    return ''.join(filter(str.isalnum, str(s))).upper()

# 약 이미지 중앙 크롭 함수
def crop_pill_area(image):
    img_cv = np.array(image)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    h, w, _ = img_cv.shape
    size = min(h, w) // 2
    center_x, center_y = w // 2, h // 2
    cropped = img_cv[center_y - size//2:center_y + size//2, center_x - size//2:center_x + size//2]
    return Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))

# 데이터 로딩
marge_all = pd.read_excel("final_data.xlsx")
for col in ['제품명', '업체명', '식별표기']:
    marge_all[col] = marge_all[col].astype(str)
    marge_all[f'{col}_clean'] = marge_all[col].apply(clean_str)

# 임산부 금기 데이터 로딩(우선 임산부만 적용용)
preg_df = pd.read_excel("pregnant_warning.xlsx")
preg_df['제품명'] = preg_df['제품명'].str.strip()

# 임산부 금기 판별 함수
def check_drug_interaction_summary(pill_list):
    warnings = []

    # 제품명 비교 정제
    preg_df['제품명_clean'] = preg_df['제품명'].str.strip().str.replace(" ", "")

    for pill in pill_list:
        product_name = pill.get("제품명", "").strip().replace(" ", "")
        match = preg_df[preg_df["제품명_clean"] == product_name]

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

# 질문 종류 판별 함수
def is_combination_question(q):
    return any(k in q for k in ["병용", "함께", "같이", "동시", "복용"])

def is_pregnancy_question(q):
    return any(k in q for k in ["임산부", "임신", "태아"])

# 세션 초기화
if 'basket' not in st.session_state:
    st.session_state['basket'] = {}
if 'displayed_candidates' not in st.session_state:
    st.session_state['displayed_candidates'] = {}
if 'user_question' not in st.session_state:
    st.session_state['user_question'] = ""

ocr_tab, select_tab, question_tab, result_tab = st.tabs([
    "① 약약 이미지 업로드",
    "② 인식 결과 & 약약 선택",
    "③ 질문 입력",
    "④ 복용 가능 여부 결과"
])

with ocr_tab:
    st.header("📷 약 이미지 업로드")
    uploaded_files = st.file_uploader("궁금하신 약의 이미지를 업로드 해주세요", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    st.session_state['ocr_texts'] = []

    if uploaded_files:
        st.markdown("🔍 인식된 식별표기를 확인 후 **본인의 약과 일치하면 '② 인식 결과 & 약 선택' 탭으로 넘어가세요.**")
        for i, uploaded_file in enumerate(uploaded_files):
            image = Image.open(uploaded_file).convert("RGB")
            cropped = crop_pill_area(image)
            st.image(cropped, caption=f"이미지 {i+1} 크롭 약 이미지", width=200)
            ocr_text = easyocr_extract_text(cropped).upper()
            edited = st.text_input(f"약 {i+1} 식별표기 (수정 가능)", value=ocr_text, key=f"ocr_edit_{i}")
            st.session_state['ocr_texts'].append(edited)

with select_tab:
    st.header("💊 인식된 약의 정보를 확인해주세요")
    displayed_candidates = {}

    if st.session_state['ocr_texts']:
        combined_identifiers = list(set([clean_str(text) for text in st.session_state['ocr_texts'] if text.strip()]))
        candidates = pd.DataFrame()
        for ident in combined_identifiers:
            temp = marge_all[marge_all['식별표기_clean'] == ident]
            if temp.empty:
                temp = marge_all[marge_all['식별표기_clean'].apply(
                    lambda x: any(difflib.get_close_matches(ident, [x], cutoff=0.6))
                )]
            candidates = pd.concat([candidates, temp])
        candidates = candidates.drop_duplicates().reset_index(drop=True)

        if not candidates.empty:
            st.markdown("📝 아래 목록 중 **복용한 약을 선택**하고 수량을 입력해 주세요.")
            for display_idx, (_, row) in enumerate(candidates.iterrows()):
                key_cb = f"select_{display_idx}"
                key_qty = f"qty_{display_idx}"
                displayed_candidates[key_cb] = row
                with st.container():
                    cols = st.columns([1, 5])
                    with cols[1]:
                        selected = st.checkbox(f"{row['식별표기']} - {row['제품명']} ({row['업체명']})", key=key_cb)
                        if selected:
                            qty = st.number_input(f"▶ {row['제품명']} 수량 선택", min_value=1, max_value=10, value=1, key=key_qty)
                            st.session_state['basket'][key_cb] = {"quantity": qty, **row.to_dict()}
                        else:
                            st.session_state['basket'].pop(key_cb, None)
                        with st.expander(" ▼ 추가 정보 보기"):
                            st.markdown(f"- **제형**: {row.get('제형', '정보 없음')}")
                            st.markdown(f"- **모양**: {row.get('모양', '정보 없음')}")
                            st.markdown(f"- **색깔**: {row.get('색깔', '정보 없음')}")
                            st.markdown(f"- **성상**: {row.get('성상', '정보 없음')}")
    st.session_state['displayed_candidates'] = displayed_candidates
    if not displayed_candidates:
        st.info("⚠️ 인식된 약이 없거나 선택하지 않았습니다. 이미지 또는 OCR 결과를 다시 확인해 주세요.")

with question_tab:
    st.header("❓ 궁금한 점을 입력해주세요")
    st.markdown("예: '이 약들을 같이 먹어도 되나요?', '임산부가 복용해도 될까요?' 등")
    st.session_state['user_question'] = st.text_input("질문 입력")
    st.markdown("✏️ 질문을 모두 작성하셨나요? 그렇다면 👉 '④ 복용 가능 여부 결과' 탭으로 이동해 주세요.")

with result_tab:
    st.header("✅ 복용 가능 여부 결과")

    displayed_candidates = st.session_state.get('displayed_candidates', {})
    basket_items = list(st.session_state['basket'].values())
    pill_infos = basket_items
    quantities = [item['quantity'] for item in basket_items]

    if basket_items and st.session_state.get('user_question'):
        q = st.session_state['user_question']
        pill_list = [deepcopy(item) for key, item in st.session_state['basket'].items() if key in displayed_candidates]

        if is_pregnancy_question(q):
            st.markdown(check_drug_interaction_summary(pill_list))
        elif is_combination_question(q):
            st.markdown("💡 약물 병용 분석 기능은 준비 중입니다.")
        else:
            for i, pill_info in enumerate(pill_list):
                st.subheader(f"약 {i+1}: {pill_info['제품명']}")
                st.markdown(f"💬 `{q}` 에 대한 응답은 준비 중입니다.")
    else:
        st.info("💬 먼저 약을 선택하고 질문을 입력해주세요.")