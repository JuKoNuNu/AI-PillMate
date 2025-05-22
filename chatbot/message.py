# ui
import streamlit as st
# 구글 권한 사용을 위한 패키지
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

# http 오류 처리용 
from googleapiclient.errors import HttpError

from datetime import datetime, timedelta

# ui
import streamlit as st
# 로그인
from streamlit_oauth import OAuth2Component
# 로그인 토큰 해석
import jwt


# 설정 : 1. openai만 사용, 2. 랭체인 에이전트를 이용 검색증강, 3. 더미
ai_res_type = 2




GOOGLE_CLIENT_ID="1017400912907-e4g3hgurjoq29h92aha1uk7uvqu4mtq2.apps.googleusercontent.com"
GOOGLE_CLIENT_SECRET="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"

def login_api():
    client_id=GOOGLE_CLIENT_ID#너 클라이언트 아이디 넣어
    client_secret=GOOGLE_CLIENT_SECRET#너 클라이언트 비번 넣어

    oauth2 = OAuth2Component(
        client_id=client_id,
        client_secret=client_secret,
        authorize_endpoint="https://accounts.google.com/o/oauth2/v2/auth",
        token_endpoint="https://oauth2.googleapis.com/token",
    )

    # 세션 상태에 token이 없으면 로그인 버튼 표시
    # 사용할 계정의 Google Calendar API를 사용 상태로 바꾸어야 사용가능

    if "token" not in st.session_state:
        token = oauth2.authorize_button(
            name="Google로 시작하기\n클릭",
            icon="",
            redirect_uri="http://localhost:8501", # 여기 나중에 로컬 아닌 버전으로 수정해야함
            scope="openid email profile https://www.googleapis.com/auth/calendar https://www.googleapis.com/auth/tasks https://www.googleapis.com/auth/drive https://www.googleapis.com/auth/spreadsheets"
        )
        if token:
            st.session_state.token = token  # 세션에 저장
            st.rerun()  # 새로고침해서 버튼 숨김
    else:
        token = st.session_state.token
        id_token = token["token"]["id_token"]
        decoded = jwt.decode(id_token, options={"verify_signature": False})

        st.success(f"✅ {decoded['name']}님 로그인됨")
        st.image(decoded['picture'], width=100)
        st.write(f"이메일: {decoded['email']}")
        if st.button("로그아웃"):
            del st.session_state["token"]
            st.rerun()







def make_creds(scope):
    client_id=GOOGLE_CLIENT_ID
    client_secret=GOOGLE_CLIENT_SECRET

    # 세션 상태에 token이 없으면 로그인 버튼 표시
    # 사용할 계정의 Google Calendar API를 사용 상태로 바꾸어야 사용가능
    if "token" not in st.session_state:
        pass
    else:
        token = st.session_state.token
        # 드라이브(스프레드시트) 사용을 위한 구글계정 정보를 세션에서 가져오기
        creds = Credentials(
            token=token["token"]["access_token"],
            refresh_token=token.get("refresh_token"),
            token_uri="https://oauth2.googleapis.com/token",
            client_id=client_id,
            client_secret=client_secret,
            scopes=[f"https://www.googleapis.com/auth/{scope}"]
        )


        return creds

def get_lgh_calendar_events(calendar_id='lyg940500@gmail.com'):
    creds = make_creds("calendar")
    
    # 미로그인시 [] 반환
    if not creds:
        st.error("❌ 먼저 로그인하세요")
        return []

    # 캘린더 API 서비스 객체 생성
    service = build("calendar", "v3", credentials=creds)

    # 캘린더에서 대충 최신 이벤트 50개 가져오기
    events_result = service.events().list(
        calendarId=calendar_id,
        timeMin="2020-01-01T00:00:00Z",
        timeMax="2030-01-01T00:00:00Z",
        maxResults=14000,#여기가 로드 값
        singleEvents=True,
        orderBy="startTime"
    ).execute()
    events = events_result.get("items", [])
    calendar_events=[]

    if not events:
        st.write("예정된 일정이 없습니다.")
    for event in events:
        
        is_datetime = "dateTime" in event["start"]
        is_summary = "summary" in event

        start = event["start"].get("dateTime", event["start"].get("date"))
        end = event.get("end", {}).get("dateTime", event.get("end", {}).get("date", None))  # end는 없을 수도 있음

        event_data = {
            "title": event['summary'] if is_summary else "제목없음",
            "start": start[:16] if is_datetime else start,
            "resourceId": "a",
            "description" : event.get("description", "설명없음"),
            "calendar_id" : calendar_id,
            "event_id" : event.get("id","아이디오류"),
            "calendar_summary" : st.session_state.calendar_list.get(calendar_id, "제목없음"),
            "all_day" : event.get("all_day", True)
        }

        if is_datetime:
            event_data["end"] = end[:16] if end else start[:16]
            event_data["allDay"] = False
        else:
            end_date = datetime.strptime(end, "%Y-%m-%d")  # 문자열 → datetime
            end_plus_one = end_date + timedelta(days=0)        # 0 더하기
            event_data["end"] = end_plus_one.strftime("%Y-%m-%d")  # 다시 문자열로 저장
            event_data["allDay"] = True

        calendar_events.append(event_data)
    return calendar_events

def session_set_calendar_list():
    creds = make_creds("calendar")

    if not creds:
        st.error("❌ 먼저 로그인하세요")
        return

    # 캘린더 API 서비스 객체 생성
    service = build("calendar", "v3", credentials=creds)
    calendar_list={}
    # 캘린더 목록 전부 가져오기
    page_token = None
    while True:
        calendar_list_origin = service.calendarList().list(pageToken=page_token).execute()
        for calendar_list_entry in calendar_list_origin['items']:
            # 캘린더 목록을 딕셔너리에 (id, summary)로 추가
            calendar_list[calendar_list_entry["id"]]=calendar_list_entry['summary'] 
        page_token = calendar_list_origin.get('nextPageToken')
        if not page_token:
            break

    st.session_state.calendar_list = calendar_list

# calendar_id의 캘린더 이벤트를 삭제 
def del_lgh_calendar_events(event_id, calendar_id='lyg940500@gmail.com'):
    creds = make_creds("calendar")
    
    # 미로그인시 반환
    if not creds:
        st.error("❌ 먼저 로그인하세요")
        return

    # 캘린더 API 서비스 객체 생성
    service = build("calendar", "v3", credentials=creds)
    
    try:
        service.events().delete(calendarId=calendar_id, eventId=event_id).execute()
        st.success("✅ 이벤트를 성공적으로 삭제했습니다.")
    except HttpError as error:
        status = error.resp.status
        if status == 404:
            st.error("❌ 이벤트를 찾을 수 없습니다. (404 Not Found)")
        elif status == 403:
            st.warning("⚠️ 해당 캘린더에 대한 권한이 없습니다. (403 Forbidden)")
        else:
            st.error(f"❌ 알 수 없는 오류가 발생했습니다: {error}")

def del_lgh_start():
    login_api()
    session=st.session_state
    if "token" in session and "calendar_list" not in session:
        session_set_calendar_list()
    if st.button("🗑️ 올 딜리트"):
        for calendar_event in get_lgh_calendar_events():
            if "event_id" in calendar_event:
                try:
                    del_lgh_calendar_events(calendar_event.get("event_id"))
                except:
                    pass