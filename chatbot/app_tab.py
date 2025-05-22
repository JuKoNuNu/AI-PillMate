# streamlit run app_tab.py


from streamlit_option_menu import option_menu
import json
import datetime
import streamlit.components.v1 as components
import os
import os.path
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
from streamlit_calendar import calendar
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
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import datetime


load_dotenv()
# print(os.environ['OPENAI_API_KEY'])
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
client = openai.OpenAI(api_key=OPENAI_API_KEY)  # 수정 예정
openai_key = os.getenv("OPENAI_API_KEY") # 수정 예정
google_key = os.getenv("GOOGLE_API_KEY")
google_cse = os.getenv("GOOGLE_CSE_ID")

# 구글 캘린더 
SCOPES = ['https://www.googleapis.com/auth/calendar.events']

def get_google_credentials():
    creds = None

    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)

    if not creds or not creds.valid or not creds.refresh_token:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES
            )
            creds = flow.run_local_server(port=8080, access_type='offline', prompt='consent')

        # 새로 발급받은 토큰 저장
        with open('token.json', 'w') as token_file:
            token_file.write(creds.to_json())

    return creds


def get_calendar_service(credentials):
    return build('calendar', 'v3', credentials=credentials)

def create_medication_event(service, drug_name, dosage_time_list, start_date, end_date, usage_instruction="없음"):
        current_date = start_date
        while current_date <= end_date:
            for time in dosage_time_list:
                hour, minute = map(int, time.split(":"))
                start_dt = datetime.datetime.combine(start_date, datetime.time(hour, minute))
                end_dt = start_dt + datetime.timedelta(minutes=30)

                event = {
                    'summary': f'💊 복약: {drug_name}',
                    'description': f'{drug_name} 드실 시간입니다.\n 복용 방법 : {usage_instruction}',
                    'start': {
                        'dateTime': start_dt.isoformat(),
                        'timeZone': 'Asia/Seoul',
                    },
                    'end': {
                        'dateTime': end_dt.isoformat(),
                        'timeZone': 'Asia/Seoul',
                    },
                    'recurrence': [
                        f'RRULE:FREQ=DAILY;UNTIL={end_date.strftime("%Y%m%d")}T000000Z'
                    ],
                    'reminders': {
                        'useDefault': False,
                        'overrides': [
                            {'method': 'popup', 'minutes': 10},
                        ],
                    },
                }
                service.events().insert(calendarId='primary', body=event).execute()



def show_calendar(service, start_date, end_date):
        time_min = datetime.datetime.combine(start_date, datetime.time.min).isoformat() + 'Z'
        time_max = datetime.datetime.combine(end_date, datetime.time.max).isoformat() + 'Z'

        events_result = service.events().list(
            calendarId='primary',
            timeMin=time_min,
            timeMax=time_max,
            singleEvents=True,
            orderBy='startTime'
        ).execute()

        events = events_result.get('items', [])

        if not events:
            st.info("📭 등록된 복약 일정이 없습니다.")
            return

        event_list = []
        for event in events:
            event_list.append({
                "title": event.get("summary", "제목 없음"),
                "start": event["start"].get("dateTime", event["start"].get("date")),
                "end": event["end"].get("dateTime", event["end"].get("date")),
                "extendedProps": {
                    "description": event.get("description", "없음")
                }
            })
        event_js_array = json.dumps(event_list, ensure_ascii=False)

        html_calendar = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset='utf-8' />
            <link href='https://cdnjs.cloudflare.com/ajax/libs/fullcalendar/6.1.8/index.global.min.css' rel='stylesheet'>
            <script src='https://cdnjs.cloudflare.com/ajax/libs/fullcalendar/6.1.8/index.global.min.js'></script>
            <style>
            #calendar {{
                max-width: 1000px;
                margin: 40px auto;
            }}
            </style>
            <script>
            document.addEventListener('DOMContentLoaded', function() {{
                var calendarEl = document.getElementById('calendar');
                var calendar = new FullCalendar.Calendar(calendarEl, {{
                    initialView: 'timeGridWeek',
                    locale: 'ko',
                    allDaySlot: false,
                    slotMinTime: "06:00:00",
                    slotMaxTime: "23:59:59",
                    height: 'auto',
                    headerToolbar: {{
                        left: 'prev,next today',
                        center: 'title',
                        right: 'dayGridMonth,timeGridWeek,timeGridDay'
                    }},
                    events: {event_js_array},
                    eventClick: function(info) {{
                        info.jsEvent.preventDefault();
                        alert(info.event.title + "\\n\\n설명: " + (info.event.extendedProps.description || "없음"));
                    }}
                }});
                calendar.render();
            }});
            </script>
        </head>
        <body>
            <div id='calendar'></div>
        </body>
        </html>
        """

        st.components.v1.html(html_calendar, height=800, scrolling=True)

def delete_medication_events(service, start_date, end_date):
    time_min = datetime.datetime.combine(start_date, datetime.time.min).isoformat() + 'Z'
    time_max = datetime.datetime.combine(end_date, datetime.time.max).isoformat() + 'Z'

    events_result = service.events().list(
        calendarId='primary',
        timeMin=time_min,
        timeMax=time_max,
        singleEvents=True,
        orderBy='startTime'
    ).execute()

    events = events_result.get('items', [])
    deleted_count = 0

    for event in events:
        if '💊 복약:' in event.get('summary', ''):
            service.events().delete(calendarId='primary', eventId=event['id']).execute()
            deleted_count += 1

    return deleted_count



# 페이지 타이틀
st.set_page_config(page_title="AI 기반 약 인식 시스템", layout="wide")
reader = easyocr.Reader(['en', 'ko'])

# OCR 
def easyocr_extract_text(image: Image.Image):
    img_array = np.array(image)
    result = reader.readtext(img_array, detail=0)
    return " ".join(result)

# 약 인식 크롭
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

# 문자열 정리
def clean_str(s):
    return ''.join(filter(str.isalnum, str(s))).upper() if isinstance(s, str) else ''

def extract_ingredients(text):
    return re.findall(r'([가-힣A-Za-z]+)\s*([\d\.]+)\s*(mg|g|㎎|밀리그램)', text)

# 챗봇 질의  
def chat_with_pill_info(pill_info, question, selected_pills=None):
    question = question.lower().strip()

    # 병용 질문 감지: selected_pills가 있고 병용 질문이면 바로 병용 분석 실행
    if selected_pills and is_combination_question(question):
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


# 병용 가능 - 약 분류 
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
            "당신은 한국어로만 대답하는 약사입니다. 항상 친절하고 상세하게 설명합니다.\n"
            "당신은 약사로서 약에 대한 정보를 사용자의 질문에 맞게 친절하고 자세하게 설명해주어야 합니다."
            "예시"
            "감기에 걸린거 같아 -> 당신은 감기에 걸렸습니다. 감기의 증상은 열, 기침, 콧물 등이 있으며 이 약 '제품명'은 이러한 증상에 효과적입니다. 사용방법은 XXX, 부작용이나 주의사항은 XXX입니다."
            "하고 진짜 약사가 충고하듯이 유사한 약도 추천해줄 수 있으면 추천해줘"
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

#------------------------------------------------------------
# 메인 함수 - 스트리밋 실행
## 메인 소개 페이지

def main_page():
    st.markdown("""
        <h1 style='text-align: center; margin-bottom: 60px;'>💊 AI 기반 다제약물(중복 복용) 예방 챗봇</h1>
        <h3 style='text-align: left; margin-top: 60px;'>안녕하세요. 비대면 개인 건강 비서입니다!</h3>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div style='
            margin-top: 30px; border: 2px solid #ccc; border-radius: 10px; padding: 20px; background-color: #f9f9f9;
        '>
                이 챗봇은 다음과 같은 기능을 제공합니다: <br><br>
            📷 <b>알약 이미지 검색</b>: 사진을 업로드하면 알약을 인식하여 복용 중복 가능 정보를 확인할 수 있어요.<br><br>
            🤖 <b>약 정보 검색 챗봇</b>: 궁금한 약 정보 질문을 하고, AI로부터 답변을 받아보세요.<br><br>
            💡 <b>기능3</b>: --------<br><br>
        </div>

        <p style='margin-top: 30px;'>▶ 왼쪽 사이드바에서 기능을 선택해주세요!</p>
    """, unsafe_allow_html=True)


## 기능 1 (ocr)

def chatbot1_page():
    st.subheader("약 이미지 인식 기반 정보 제공 시스템")

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



## 기능 2 (챗봇)
def chatbot2_page():
    st.subheader("💊 AI 기반 약품 추천 챗봇")
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

## 기능 3 
def chatbot3_page():
    st.subheader("약 이미지 인식 기반 캘린더 등록")
    st.write("--설명--")
    if "google_credentials" not in st.session_state:
        st.session_state["google_credentials"] = None
    if "basket" not in st.session_state:
        st.session_state["basket"] = {}

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

        if st.session_state.get('basket'):
            st.subheader("3️⃣ 복약 일정 설정")

            dosage_times = st.multiselect("복용 시간 선택", ["06:00", "09:00", "12:00", "15:00", "18:00", "21:00", "24:00"])
            start_date = st.date_input("시작일")
            end_date = st.date_input("종료일")

            if st.button("🔐 로그인 다시 시도"):
                if os.path.exists("token.json"):
                    os.remove("token.json")
                    st.success("✅ 기존 로그인 정보 삭제 완료. 다시 로그인 해주세요.")

            if st.button("🗑️ 기존 복약 일정 삭제"):
                try:
                    creds = st.session_state.get('google_credentials', get_google_credentials())
                    service = get_calendar_service(creds)

                    deleted = delete_medication_events(service, start_date, end_date)
                    if deleted > 0:
                        st.success(f"🗑️ 총 {deleted}개의 복약 일정이 삭제되었습니다.")
                    else:
                        st.info("📭 삭제할 복약 일정이 없습니다.")
                except Exception as e:
                    st.error(f"❌ 일정 삭제 중 오류 발생: {e}")

            if st.button("📅 복약 일정 등록"):
                if not dosage_times:
                    st.warning("복약 시간을 선택해주세요.")
                else:
                    try:
                        creds = get_google_credentials()
                        st.session_state['google_credentials'] = creds
                        service = get_calendar_service(creds)

                        excel_data = load_excel_data()  # 복용 방법 추출용 엑셀 불러오기

                        for key in st.session_state['basket']:
                            drug_info = displayed_candidates[key]
                            qty = st.session_state['basket'][key]["quantity"]
                            drug_name = drug_info['제품명']

                            # 복용 방법 추출
                            usage_instruction = "없음"
                            row = excel_data[excel_data["제품명"] == drug_name]
                            if not row.empty:
                                usage_instruction = row.iloc[0].get("용법용량", "없음")

                            # 일정 등록
                            create_medication_event(
                                service,
                                drug_name,
                                dosage_times,
                                start_date,
                                end_date,
                                usage_instruction
                            )

                        st.success("✅ 복약 일정이 Google 캘린더에 등록되었습니다.")
                        show_calendar(service, start_date, end_date)

                    except Exception as e:
                        st.error(f"❌ 로그인 실패 또는 캘린더 등록 실패: {e}")

            st.subheader("💬 AI 복약 상담")

            if "question_history" not in st.session_state:
                st.session_state["question_history"] = []

            question = st.text_area("궁금한 점을 입력해 주세요 (예: 같이 먹어도 되나요?, 보관 방법은?)", height=100)

            if st.button("💬 AI 질문하기"):
                if not question.strip():
                    st.warning("질문을 입력해주세요.")
                else:
                    st.header("💬 AI 복약 상담 결과")

                    basket_items = [displayed_candidates[key] for key in st.session_state['basket']]
                    quantities = [st.session_state['basket'][key]["quantity"] for key in st.session_state['basket']]

                    answer_blocks = []

                    if is_combination_question(question):
                        interaction_summary = check_drug_interaction_summary(basket_items)
                        dosage_check = check_ingredient_overlap_and_dosage(basket_items, quantities)

                        st.markdown(interaction_summary)
                        st.markdown(dosage_check)

                        # 결과 저장
                        full_answer = interaction_summary + "\n\n" + dosage_check
                        st.session_state["question_history"].append({
                            "question": question,
                            "answer": full_answer
                        })
                    else:
                        full_answer = ""
                        for i, pill_info in enumerate(basket_items):
                            st.subheader(f"약 {i+1}: {pill_info['제품명']}")
                            answer = chat_with_pill_info(pill_info, question)
                            st.write(answer)
                            full_answer += f"약 {i+1} ({pill_info['제품명']}): {answer}\n\n"

                        interaction_summary = check_drug_interaction_summary(basket_items)
                        dosage_check = check_ingredient_overlap_and_dosage(basket_items, quantities)

                        st.markdown(interaction_summary)
                        st.markdown(dosage_check)

                        full_answer += interaction_summary + "\n\n" + dosage_check

                        # 결과 저장
                        st.session_state["question_history"].append({
                            "question": question,
                            "answer": full_answer
                        })

            # 이전 질문 보기 (질문 + 답변)
            if st.session_state["question_history"]:
                with st.expander("📝 이전 질문 보기"):
                    for i, qa in enumerate(reversed(st.session_state["question_history"]), 1):
                        st.markdown(f"**{i}. 질문:** {qa['question']}")
                        st.markdown(f"**답변:** {qa['answer']}")
                        st.markdown("---")
             
# UI 
def main():
    with st.sidebar:
        page = option_menu(
            "MENU",
            ["소개", "알약 이미지 검색", "약 정보 검색 챗봇", "--기능3--"],
            icons=["capsule", "camera", "robot", "cast"],
            default_index=0,
            styles={
                "container": {"padding": "5px", "background-color": "#f0f0f0"},
                "icon": {"color": "black", "font-size": "20px"},
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px"},
                "nav-link-selected": {"background-color": "#9a9a9a", "color": "white"},
            }
        )

    # 라우팅
    if page == "소개":
        main_page()
    elif page == "알약 이미지 검색":
        chatbot1_page()
    elif page == "약 정보 검색 챗봇":
        chatbot2_page()
    elif page == "--기능3--":
        chatbot3_page()

# 실행
if __name__ == "__main__":
    main()
 # type: ignore
