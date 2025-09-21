"""
구글 캘린더 서비스 모듈
"""
import os
import json
import datetime
from typing import List, Optional
import streamlit as st
import streamlit.components.v1 as components
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from config.settings import Config


class CalendarService:
    """구글 캘린더 서비스 클래스"""
    
    def __init__(self):
        self.config = Config()
        self.service = None
        self.credentials = None
    
    def get_credentials(self) -> Credentials:
        """Google 인증 정보 획득"""
        creds = None
        
        if os.path.exists(self.config.TOKEN_FILE):
            creds = Credentials.from_authorized_user_file(
                self.config.TOKEN_FILE, self.config.CALENDAR_SCOPES
            )
        
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.config.CREDENTIALS_FILE, self.config.CALENDAR_SCOPES
                )
                creds = flow.run_local_server(
                    port=8080, access_type='offline', prompt='consent'
                )
            
            # 토큰 저장
            with open(self.config.TOKEN_FILE, 'w') as token_file:
                token_file.write(creds.to_json())
        
        self.credentials = creds
        return creds
    
    def get_service(self):
        """구글 캘린더 서비스 객체 획득"""
        if not self.credentials:
            self.get_credentials()
        
        self.service = build('calendar', 'v3', credentials=self.credentials)
        return self.service
    
    def create_medication_event(
        self, 
        drug_name: str, 
        dosage_times: List[str], 
        start_date: datetime.date, 
        end_date: datetime.date, 
        usage_instruction: str = "없음"
    ) -> bool:
        """복약 일정 이벤트 생성"""
        try:
            if not self.service:
                self.get_service()
            
            first_time = dosage_times[0]
            h0, m0 = map(int, first_time.split(":"))
            start_dt = datetime.datetime.combine(start_date, datetime.time(h0, m0))
            end_dt = start_dt + datetime.timedelta(minutes=30)
            
            hours = ",".join(str(int(t.split(":")[0])) for t in dosage_times)
            minutes = ",".join(str(int(t.split(":")[1])) for t in dosage_times)
            
            rrule = (
                f"RRULE:FREQ=DAILY;"
                f"UNTIL={end_date.strftime('%Y%m%d')}T235959Z;"
                f"BYHOUR={hours};"
                f"BYMINUTE={minutes}"
            )
            
            event = {
                'summary': f'💊 복약: {drug_name}',
                'description': f'{drug_name} 드실 시간입니다.\n복용 방법: {usage_instruction}',
                'start': {'dateTime': start_dt.isoformat(), 'timeZone': 'Asia/Seoul'},
                'end': {'dateTime': end_dt.isoformat(), 'timeZone': 'Asia/Seoul'},
                'recurrence': [rrule],
                'reminders': {
                    'useDefault': False, 
                    'overrides': [{'method': 'popup', 'minutes': 10}]
                }
            }
            
            self.service.events().insert(calendarId='primary', body=event).execute()
            return True
            
        except Exception as e:
            st.error(f"캘린더 이벤트 생성 실패: {e}")
            return False
    
    def delete_medication_events(
        self, 
        start_date: datetime.date, 
        end_date: datetime.date
    ) -> int:
        """복약 일정 삭제"""
        try:
            if not self.service:
                self.get_service()
            
            time_min = datetime.datetime.combine(start_date, datetime.time.min).isoformat() + 'Z'
            time_max = datetime.datetime.combine(end_date, datetime.time.max).isoformat() + 'Z'
            
            deleted_count = 0
            page_token = None
            
            while True:
                events_result = self.service.events().list(
                    calendarId='primary',
                    timeMin=time_min,
                    timeMax=time_max,
                    singleEvents=True,
                    orderBy='startTime',
                    maxResults=250,
                    pageToken=page_token
                ).execute()
                
                events = events_result.get('items', [])
                
                for event in events:
                    if '💊 복약:' in event.get('summary', ''):
                        self.service.events().delete(
                            calendarId='primary',
                            eventId=event['id']
                        ).execute()
                        deleted_count += 1
                
                page_token = events_result.get('nextPageToken')
                if not page_token:
                    break
            
            return deleted_count
            
        except Exception as e:
            st.error(f"캘린더 이벤트 삭제 실패: {e}")
            return 0
    
    def show_calendar(self, start_date: datetime.date, end_date: datetime.date):
        """캘린더 표시"""
        try:
            if not self.service:
                self.get_service()
            
            time_min = datetime.datetime.combine(start_date, datetime.time.min).isoformat() + 'Z'
            time_max = datetime.datetime.combine(end_date, datetime.time.max).isoformat() + 'Z'
            
            events_result = self.service.events().list(
                calendarId='primary',
                timeMin=time_min,
                timeMax=time_max,
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            
            events = events_result.get('items', [])
            
        except Exception as e:
            st.error(f"Google Calendar API 호출 실패: {e}")
            return
        
        if not events:
            st.info("등록된 복약 일정이 없습니다.")
            return
        
        event_list = []
        for event in events:
            try:
                start = event["start"].get("dateTime") or event["start"].get("date")
                end = event["end"].get("dateTime") or event["end"].get("date")
                if not start or not end:
                    continue
                
                event_list.append({
                    "title": event.get("summary", "제목 없음"),
                    "start": start,
                    "end": end,
                    "extendedProps": {
                        "description": event.get("description", "없음")
                    }
                })
            except Exception as e:
                st.warning(f"이벤트 파싱 중 오류: {e}")
        
        try:
            event_js_array = json.dumps(event_list[:100], ensure_ascii=False)
        except Exception as e:
            st.error(f"이벤트 JSON 변환 실패: {e}")
            return
        
        self._render_calendar_html(event_js_array)
    
    def _render_calendar_html(self, event_js_array: str):
        """캘린더 HTML 렌더링"""
        html_calendar = f"""
        <!DOCTYPE html>
        <html lang="ko">
        <head>
            <meta charset='utf-8' />
            <link href='https://cdnjs.cloudflare.com/ajax/libs/fullcalendar/6.1.8/index.global.min.css' rel='stylesheet'>
            <script src='https://cdnjs.cloudflare.com/ajax/libs/fullcalendar/6.1.8/index.global.min.js'></script>
            <script src="https://unpkg.com/@popperjs/core@2"></script>
            <script src="https://unpkg.com/tippy.js@6"></script>
            <link rel="stylesheet" href="https://unpkg.com/tippy.js@6/animations/scale.css" />
            <style>
                body {{
                    font-family: 'Apple SD Gothic Neo', 'Segoe UI', sans-serif;
                    margin: 0; padding: 0; background-color: #f9fafb;
                }}
                .fc {{
                    max-width: 1000px; margin: 0 auto; padding: 20px;
                    background-color: #ffffff; border-radius: 12px;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
                }}
                .fc-toolbar-title {{
                    font-size: 1.6em; font-weight: bold; color: #1f2937;
                }}
                .fc-button {{
                    background-color: #ffffff !important;
                    border: 1px solid #d1d5db !important;
                    color: #374151 !important; border-radius: 6px !important;
                    font-size: 14px !important; padding: 5px 12px !important;
                    box-shadow: none !important; transition: all 0.2s ease;
                }}
                .fc-button:hover {{ background-color: #f3f4f6 !important; }}
                .fc-button-active, .fc-button:active {{
                    background-color: #e0f2fe !important;
                    color: #2563eb !important; border-color: #bfdbfe !important;
                }}
            </style>
            <script>
                document.addEventListener('DOMContentLoaded', function() {{
                    var calendarEl = document.getElementById('calendar');
                    var calendar = new FullCalendar.Calendar(calendarEl, {{
                        initialView: 'dayGridMonth',
                        locale: 'ko',
                        headerToolbar: {{
                            left: 'prev,next today',
                            center: 'title',
                            right: 'dayGridMonth,timeGridWeek,timeGridDay'
                        }},
                        buttonText: {{
                            today: '오늘', month: '월간', week: '주간',
                            day: '일간', list: '목록'
                        }},
                        events: {event_js_array},
                        eventDidMount: function(info) {{
                            tippy(info.el, {{
                                content: info.event.extendedProps.description || "없음",
                                placement: 'top', animation: 'scale', theme: 'light-border'
                            }});
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
        
        try:
            components.html(html_calendar, height=850, scrolling=True)
        except Exception as e:
            st.error(f"HTML 렌더링 실패: {e}")