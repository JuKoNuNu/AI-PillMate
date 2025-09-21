import os
import re
import requests
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
from PIL import Image
from openai import OpenAI as OpenAIClient
from llama_index.core import VectorStoreIndex, Document, StorageContext, load_index_from_storage, Settings
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever


load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
google_key = os.getenv("GOOGLE_API_KEY")
google_cse = os.getenv("GOOGLE_CSE_ID")
naver_client_id = os.getenv("NAVER_API_ID")
naver_client_secret = os.getenv("NAVER_API_SECRET")

client = OpenAIClient(api_key=openai_key)
os.environ["OPENAI_API_KEY"] = openai_key


# 엑셀 데이터 로딩
@st.cache_data
def load_excel_data():
    return pd.read_excel("txta.xlsx")

df = load_excel_data()

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
            "먼저 다음 형식으로 출력하세요:\n"
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
    retriever = VectorIndexRetriever(index=index, similarity_top_k=5)
    response_synthesizer = get_response_synthesizer(response_mode="tree_summarize")

    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer
    )

    return query_engine

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

# 챗봇 페이지 구현
def chatbot_page():
    st.title("💊 AI 기반 약품 추천 챗봇")
    st.markdown("사용자 질문에 따라 약 정보를 추천합니다.")

    mode = st.radio("방식 선택", ["기존 방식 (키워드 기반)", "RAG 기반 (LlamaIndex)"])
    user_question = st.text_input("❓ 증상이나 궁금한 약을 입력하세요")

    if user_question:
        if mode == "기존 방식 (키워드 기반)":
            st.markdown("🔍 **GPT 키워드 기반 약 검색 중...**")
            keyword = extract_keyword(user_question)
            st.write("🔑 추출된 키워드:", keyword)
            results = find_related_drugs(df, keyword)


            if not results.empty:
                for _, row in results.iterrows():

                    name = row['제품명']
                    excel_img = row['저장_이미지_파일명']
                    local_img_path = os.path.join("images", excel_img)

                    st.markdown(f"### 💊 {row['제품명']} ({row['구분']})")
                    local_img_path = os.path.join("images", row['저장_이미지_파일명'])
                    if os.path.isfile(local_img_path):
                        st.image(local_img_path, width=300)
                    google_img_url = google_image_search(google_key, google_cse, name, num=1)
                    if google_img_url:
                        st.image(google_img_url[0], caption=f"{name} (검색 이미지)", use_container_width=300)
                    else:
                        st.info("🔍 구글 이미지 검색 결과 없음")
                    st.markdown(f"### 💊 {name} ({row['구분']})")
                    st.markdown(f"**효능효과:** {row['효능효과']}")
                    if row['사용시주의사항']:
                        with st.expander("📌 사용시 주의사항 보기"):
                            st.markdown(row['사용시주의사항'])
                    else:
                        st.info("⚠️ 주의사항 정보가 없습니다.")

                    st.markdown("---")
            else:
                st.warning(f"❌ '{keyword}' 또는 '{user_question}' 관련 약품을 찾지 못했어요.")

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
                st.markdown(f"### 💊 {product_name} ({product_class})")
                st.markdown(f"**AI 답변:** {answer_text}")

                # 🔹 로컬 이미지 출력
                excel_row = df[df["제품명"].str.lower() == product_name.lower()]
                if not excel_row.empty:
                    local_img = excel_row.iloc[0].get("저장_이미지_파일명", "")
                    local_img_path = os.path.join("images", local_img)
                    if os.path.isfile(local_img_path):
                        st.image(local_img_path, width=300)

                # 🔹 구글 이미지 출력
                google_img_url = google_image_search(google_key, google_cse, product_name, num=1)
                if google_img_url:
                    st.image(google_img_url[0], caption=f"{product_name} (검색 이미지)", width=300)
                else:
                    st.info("🔍 구글 이미지 검색 결과 없음")

                # 🔹 효능효과 출력
                st.markdown(f"**효능효과:** {effect_text}")

                # 🔹 주의사항 출력
                if not excel_row.empty:
                    warning = excel_row.iloc[0].get("사용시주의사항", "")
                    if isinstance(warning, str) and warning.strip():
                        with st.expander("📌 사용시 주의사항 보기"):
                            st.markdown(warning)
                    else:
                        st.info("⚠️ 주의사항 정보가 없습니다.")
                
                # 🔹 추가 설명 출력
                
                
            
            else:
                st.warning("❗ 제품명을 찾지 못했습니다.")

def main():
    chatbot_page()

if __name__ == "__main__":
    main()
