import os
import re
import json
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

from langchain_ollama import ChatOllama, OllamaEmbeddings

# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Dict

load_dotenv()

GMS_BASE_URL = "https://gms.ssafy.io/gmsapi/api.openai.com/v1/"
WIKI_API_URL = "https://ko.wikipedia.org/w/api.php"
CONTACT = os.getenv("WIKI_CONTACT", "").strip()
USER_AGENT = f"FitnessLab/0.1 ({CONTACT}) python-requests" if CONTACT else "FitnessLab/0.1 python-requests"
CITE_PATTERN = re.compile(r"\[근거\s*([\d,\s]+)\]")

# 한글 매핑 딕셔너리 (조회용)
BODYPART_MAP = {'Chest': '가슴', 'Middle Back': '등 중앙', 'Lower Back': '하배부', 'Lats': '광배근', 'Biceps': '이두근', 'Triceps': '삼두근', 'Shoulders': '어깨', 'Abdominals': '복근', 'Glutes': '둔근', 'Quadriceps': '앞벅지', 'Hamstrings': '뒷벅지', 'Calves': '종아리', 'Neck': '목', 'Traps': '승모근', 'Forearms': '전완근'}
EQUIPMENT_MAP = {'Barbell': '바벨', 'Dumbbell': '덤벨', 'Cable': '케이블', 'Machine': '머신', 'Body Only': '맨몸', 'Bands': '밴드', 'Kettlebells': '케틀벨', 'None': '맨몸'}
LEVEL_MAP = {'Beginner': '초급', 'Intermediate': '중급', 'Expert': '고급'}

# --- 1. 초기 설정 및 모델 로드 ---
# @st.cache_resource
# def get_model() -> ChatOpenAI:
#     key = os.environ.get("GMS_KEY")
#     if not key:
#         st.error("GMS_KEY가 없습니다. .env를 확인하세요.")
#         st.stop()
#     return ChatOpenAI(model="gpt-6.1-sol", api_key=key, base_url=GMS_BASE_URL)

# @st.cache_resource
# def get_vectorstore():
#     key = os.environ.get("GMS_KEY")
#     embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=key, base_url=GMS_BASE_URL)
#     if not os.path.exists("./chroma_meta_db"):
#         st.error("메타데이터 색인이 없습니다. 1_index_meta.py를 먼저 실행하세요.")
#         st.stop()
#     return Chroma(persist_directory="./chroma_meta_db", embedding_function=embeddings, collection_name="exercise_meta")

@st.cache_data
def load_csv_data():
    """운동 둘러보기 화면에서 사용할 CSV 데이터 로드"""
    df = pd.read_csv("dataset/gym_exercises.csv")
    # 결측값(NaN)뿐 아니라 빈 문자열과 공백만 있는 셀도 결측으로 처리합니다.
    # 색인 스크립트(1_index_meta.py)와 동일하게 전체 컬럼에 결측이 있는 행을 제외합니다.
    df = df.replace(r"^\s*$", pd.NA, regex=True).dropna().copy()
    for column in ["Title", "BodyPart", "Level", "Equipment", "Desc"]:
        df[column] = df[column].astype(str).str.strip()
    # CSV 원본에 빈 값과 다르게 저장된 공백도 최종적으로 제외합니다.
    df = df[df["Desc"].ne("")].copy()
    df['BodyPart_KR'] = df['BodyPart'].map(lambda x: BODYPART_MAP.get(x, x))
    df['Equipment_KR'] = df['Equipment'].map(lambda x: EQUIPMENT_MAP.get(x, x))
    df['Level_KR'] = df['Level'].map(lambda x: LEVEL_MAP.get(x, x))
    return df

LOCAL_DB_DIR = "./chroma_local_app"
LOCAL_COLLECTION = "exercise_meta_local"

@st.cache_resource
def get_model() -> ChatOllama:
    return ChatOllama(
        model="qwen3:8b-q4_K_M",
        base_url="http://localhost:11434",
        temperature=0,
        reasoning=False,
        seed=42,
        num_ctx=8192,
        num_predict=1024,
        keep_alive="30m",
    )

@st.cache_resource
def get_embeddings():
    return OllamaEmbeddings(
        model="bge-m3",
        base_url="http://localhost:11434",
    )

@st.cache_resource
def get_vectorstore():
    if not os.path.exists(LOCAL_DB_DIR):
        st.error("로컬 운동 색인이 없습니다. 로컬 설정으로 색인을 생성하세요.")
        st.stop()
    return Chroma(
        persist_directory=LOCAL_DB_DIR,
        embedding_function=get_embeddings(),
        collection_name=LOCAL_COLLECTION,
    )


# --- 2. 외부 문서 수집 (위키백과 API) ---
def search_wiki(query: str, limit: int = 3) -> list:
    params = {"action": "query", "list": "search", "srsearch": query, "srlimit": limit, "format": "json", "formatversion": 2}
    resp = requests.get(WIKI_API_URL, params=params, headers={"User-Agent": USER_AGENT})
    return resp.json()["query"]["search"] if resp.status_code == 200 else []

def fetch_wiki_page(title: str) -> dict | None:
    params = {"action": "query", "prop": "extracts|info", "titles": title, "explaintext": 1, "inprop": "url", "redirects": 1, "format": "json", "formatversion": 2}
    resp = requests.get(WIKI_API_URL, params=params, headers={"User-Agent": USER_AGENT})
    if resp.status_code == 200:
        page = resp.json()["query"]["pages"][0]
        if not page.get("missing"):
            return {"title": page["title"], "url": page["fullurl"], "text": page.get("extract", "")}
    return None

# --- 3. 한국어 질의 해석 / 조건 기반 추천 / 일반 지식 RAG ---
# CSV 원본은 영어입니다. 질문의 한국어 조건만 영어 컬럼 값으로 정규화합니다.
BACK_PARTS = ["Lats", "Middle Back", "Lower Back"]
BODY_PART_ALIASES = {
    "가슴": ["Chest"], "흉근": ["Chest"], "등": BACK_PARTS,
    "광배": ["Lats"], "등 중앙": ["Middle Back"], "허리": ["Lower Back"],
    "어깨": ["Shoulders"], "삼각근": ["Shoulders"],
    "이두": ["Biceps"], "삼두": ["Triceps"],
    "전완": ["Forearms"], "복근": ["Abdominals"], "코어": ["Abdominals"],
    "엉덩이": ["Glutes"], "둔근": ["Glutes"],
    "앞벅지": ["Quadriceps"], "허벅지 앞": ["Quadriceps"],
    "뒷벅지": ["Hamstrings"], "허벅지 뒤": ["Hamstrings"],
    "종아리": ["Calves"], "승모근": ["Traps"], "목": ["Neck"],
}
LEVEL_ALIASES = {
    "Beginner": ["초보", "입문", "초급", "헬린이", "beginner"],
    "Intermediate": ["중급", "intermediate"],
    "Expert": ["고급", "상급", "숙련", "expert"],
}
EQUIPMENT_ALIASES = {
    "Barbell": ["바벨", "barbell"], "Dumbbell": ["덤벨", "아령", "dumbbell"],
    "Cable": ["케이블", "cable"], "Machine": ["머신", "machine"],
    "Body Only": ["맨몸", "기구 없이", "기구없는", "bodyweight"],
    "Bands": ["밴드", "resistance band"], "Kettlebells": ["케틀벨", "kettlebell"],
}


def fallback_query_analysis(question: str) -> dict:
    """LLM 응답이 실패해도 대표적인 한국어 질문을 처리합니다."""
    q = question.lower()
    parts = []
    for alias, english_parts in BODY_PART_ALIASES.items():
        if alias == "등":
            matched = re.search(r"등(?=\s|운동|근육|을|이|에|쪽|$)", q) is not None
        else:
            matched = alias in q
        if matched:
            parts.extend(english_parts)

    level = next((k for k, aliases in LEVEL_ALIASES.items() if any(w in q for w in aliases)), None)
    equipment = next((k for k, aliases in EQUIPMENT_ALIASES.items() if any(w in q for w in aliases)), None)
    intent_markers = ("추천", "찾아", "골라", "목록", "보여줘", "어떤 운동", "뭐가 좋")
    is_recommend = any(s in q for s in intent_markers) and ("운동" in q or "추천" in q)
    return {
        "intent": "recommend" if is_recommend else "knowledge",
        "body_parts": list(dict.fromkeys(parts)),
        "level": level,
        "equipment": equipment,
        "english_query": question,
        "wiki_keyword_ko": "",
    }


def analyze_user_question(question: str) -> dict:
    """자연어를 분류하고, 영어 원본 CSV 필터 및 영어 임베딩 검색어로 변환합니다."""
    fallback = fallback_query_analysis(question)
    prompt = f"""운동 데이터베이스 질의 분석기입니다. 아래 사용자 질문을 다음 JSON 형태로만 변환하세요.
반드시 JSON 객체 하나만 출력하고 코드블록/부연 설명을 넣지 마세요.

질문: {question}

규칙:
1) intent: 운동 종목 추천/목록 검색이라면 "recommend", 자세·효과·주의사항처럼
   운동 지식을 묻는다면 "knowledge".
2) body_parts: 영문 CSV BodyPart 값의 배열. 허용값:
   Chest, Middle Back, Lower Back, Lats, Biceps, Triceps, Shoulders,
   Abdominals, Glutes, Quadriceps, Hamstrings, Calves, Neck, Traps, Forearms,
   Adductors, Abductors.
   '등 운동'은 ["Lats", "Middle Back", "Lower Back"]로 해석하세요.
   부위 조건이 없으면 빈 배열([])로 출력하세요.
3) level: "Beginner", "Intermediate", "Expert" 중 명시된 값만 사용하고,
   명시되지 않으면 null. '초보자'는 "Beginner".
4) equipment: "Barbell", "Dumbbell", "Cable", "Machine", "Body Only",
   "Bands", "Kettlebells" 중 하나. 지정되지 않으면 null.
5) english_query: 질문을 영어 운동 용어로 자연스럽게 번역한 검색 문장.
6) wiki_keyword_ko: 한국어 위키백과에서 검색할 핵심 운동/해부학 용어 하나.

예시 JSON:
{{"intent":"recommend","body_parts":["Lats","Middle Back","Lower Back"],
"level":"Beginner","equipment":null,"english_query":"beginner back exercises",
"wiki_keyword_ko":"등 근육"}}"""
    try:
        response = get_model().invoke(prompt).content
        # 모델이 JSON 앞뒤에 예기치 않은 문자를 반환하더라도 JSON만 뽑습니다.
        raw = response[response.index("{"):response.rindex("}") + 1]
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            return fallback

        allowed_parts = set(BODYPART_MAP)
        raw_parts = parsed.get("body_parts", [])
        if not isinstance(raw_parts, list):
            raw_parts = []
        model_parts = [p for p in raw_parts if isinstance(p, str) and p in allowed_parts]
        # 대표적인 한국어 조건은 규칙 기반으로 보강하여 누락을 방지합니다.
        parts = list(dict.fromkeys(fallback["body_parts"] + model_parts))

        allowed_levels = set(LEVEL_ALIASES)
        allowed_equipment = set(EQUIPMENT_ALIASES)
        return {
            "intent": "recommend" if fallback["intent"] == "recommend"
                or parsed.get("intent") == "recommend" else "knowledge",
            "body_parts": parts,
            "level": fallback["level"] or (
                parsed.get("level") if parsed.get("level") in allowed_levels else None),
            "equipment": fallback["equipment"] or (
                parsed.get("equipment") if parsed.get("equipment") in allowed_equipment else None),
            "english_query": (str(parsed.get("english_query") or question)).strip(),
            "wiki_keyword_ko": (str(parsed.get("wiki_keyword_ko") or "")).strip(),
        }
    except (ValueError, TypeError, KeyError, AttributeError):
        return fallback


def find_matching_exercises(analysis: dict, limit: int = 6) -> list[dict]:
    """유사도 검색이 아니라 영어 CSV 원본의 실제 속성으로 정확히 필터링합니다."""
    filtered = load_csv_data()
    if analysis["body_parts"]:
        filtered = filtered[
            filtered["BodyPart"].astype(str).str.strip().isin(analysis["body_parts"])
        ]
    if analysis["level"]:
        filtered = filtered[
            filtered["Level"].astype(str).str.strip() == analysis["level"]
        ]
    if analysis["equipment"]:
        selected_equipment = analysis["equipment"]
        valid_equipment = {selected_equipment}
        if selected_equipment == "Body Only":
            valid_equipment.add("None")
        filtered = filtered[
            filtered["Equipment"].astype(str).str.strip().isin(valid_equipment)
        ]

    # 한 부위에만 결과가 쏠리지 않도록 부위별로 번갈아 뽑습니다.
    filtered = filtered.drop_duplicates(subset=["Title"]).sort_values("Title")
    groups = {p: g.to_dict("records") for p, g in filtered.groupby("BodyPart", sort=True)}
    selected = []
    while len(selected) < limit and any(groups.values()):
        for part in groups:
            if groups[part] and len(selected) < limit:
                selected.append(groups[part].pop(0))
    return selected


def make_exercise_sources(rows: list[dict]) -> list[dict]:
    sources = []
    for row in rows:
        title = str(row["Title"]).strip()
        part = str(row["BodyPart"]).strip()
        level = str(row["Level"]).strip()
        equipment = str(row["Equipment"]).strip()
        desc = str(row["Desc"]).strip()
        sources.append({
            "title": title,
            "url": "",  # 로컬 CSV이므로 실제 웹 URL은 없습니다.
            "type": "운동 데이터셋",
            "text": (f"[운동명] {title}\n[자극 부위] {part}\n"
                     f"[난이도] {level}\n[필요 기구] {equipment}\n[설명] {desc}"),
        })
    return sources


class RAGState(TypedDict):
    question: str
    sources: List[Dict]
    context: str
    answer: str
    analysis: dict


def retrieve_node(state: RAGState) -> dict:
    llm = get_model()
    analysis = analyze_user_question(state["question"])
    sources = []

    if analysis["intent"] == "recommend":
        # 운동 종목 추천 질문은 영문 메타데이터의 조건으로 먼저 검색합니다.
        rows = find_matching_exercises(analysis)
        sources = make_exercise_sources(rows)
    else:
        # 자세·효과 등 일반 질문은 영어 설명과 운동명을 찾도록 영어 검색어 사용.
        db = get_vectorstore()
        meta_docs = db.similarity_search(analysis["english_query"], k=5)
        for doc in meta_docs:
            sources.append({
                "title": doc.metadata.get("title", "운동 DB"),
                "url": "",
                "text": doc.page_content,
                "type": "메타데이터",
            })

        # 한국어 위키백과 검색은 한국어 핵심어로 별도 수행합니다.
        keyword = analysis["wiki_keyword_ko"]
        if not keyword:
            extract_prompt = (
                f"다음 질문과 관련된 한국어 위키백과 검색어 1개를 한국어 명사로만 출력하세요: "
                f"{state['question']}"
            )
            keyword = str(llm.invoke(extract_prompt).content).strip()

        try:
            wiki_results = search_wiki(keyword)
            if wiki_results:
                page_data = fetch_wiki_page(wiki_results[0]["title"])
                if page_data and page_data["text"].strip():
                    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
                    chunks = splitter.split_text(page_data["text"])
                    if chunks:
                        wiki_db = Chroma.from_texts(
                                    texts=chunks,
                                    embedding=get_embeddings(),
                                )

                        for chunk in wiki_db.similarity_search(state["question"], k=2):
                            sources.append({
                                "title": page_data["title"],
                                "url": page_data["url"],
                                "text": chunk.page_content,
                                "type": "위키백과",
                            })
        except (requests.RequestException, KeyError, ValueError):
            # 위키백과가 일시적으로 실패해도 내부 운동 DB 기반 답변은 진행합니다.
            pass

    context = "\n\n".join(
        f"[근거 {i}] {s['title']}\n{s['text']}" for i, s in enumerate(sources, start=1)
    )
    return {"sources": sources, "context": context, "analysis": analysis}


def generate_node(state: RAGState) -> dict:
    sources = state["sources"]
    analysis = state["analysis"]
    if not sources:
        return {"answer": (
            "죄송합니다. 요청하신 조건에 맞는 운동을 현재 데이터셋에서 찾지 못했습니다. "
            "부위, 난이도 또는 기구 조건을 조금 넓혀서 다시 질문해 주세요."
        )}

    if analysis["intent"] == "recommend":
        prompt = f"""당신은 한국어로 응답하는 전문 피트니스 코치입니다.
사용자가 요청한 조건을 만족하는 운동을 데이터셋에서 이미 정확히 찾았습니다.
아래 [추천 후보]만 사용하여 사용자의 질문에 직접 답하세요.
- 반드시 자연스러운 한국어 존댓말(~합니다/~하세요)을 사용하세요.
- 4~6개 운동을 추천하되 후보가 적으면 있는 만큼만 소개하세요.
- 운동명은 한국 헬스장에서 쓰는 자연스러운 한글 이름으로 표기하고
  정확한 데이터 조회를 위해 원본 영문 운동명도 괄호 안에 병기하세요.
- 각 운동의 추천 이유는 주어진 BodyPart/Level/Equipment/Desc를 활용하세요.
- 각 운동 바로 뒤나 설명 끝에 후보의 [근거 N]을 정확히 인용하세요.
- 목록 밖의 운동, 확인되지 않은 효과·주의사항을 지어내지 마세요.
- '문서에서 확인되지 않는다'처럼 답변을 회피하지 말고 실제 후보를 추천하세요.
- 후보가 포함된 것만으로 부상을 예방하거나 치료할 수 있다고 단정하지 마세요.

[추천 후보]\n{state['context']}\n\n[질문] {state['question']}"""
    else:
        prompt = f"""당신은 한국어 존댓말로 답변하는 전문 피트니스 트레이너입니다.
- 아래 [근거 문서]에서 실제로 확인 가능한 내용만 사용하세요.
- 질문에 직접 답하고, 확인된 사실 뒤에는 [근거 N]을 표시하세요.
- 영어 운동명은 자연스러운 한글 헬스장 용어로 쓰되 필요하면 원문을 병기하세요.
- 근거에 없는 부분만 '제공된 자료만으로는 확인할 수 없습니다'라고 밝혀주세요.
- 근거가 있는 내용까지 무조건 답변을 거절하지 마세요.
- 위험한 증상이나 부상은 진단하지 말고 전문가 상담을 권하세요.

[근거 문서]\n{state['context']}\n\n[질문] {state['question']}"""
    return {"answer": get_model().invoke(prompt).content}


graph_builder = StateGraph(RAGState)
graph_builder.add_node("retrieve", retrieve_node)
graph_builder.add_node("generate", generate_node)
graph_builder.add_edge(START, "retrieve")
graph_builder.add_edge("retrieve", "generate")
graph_builder.add_edge("generate", END)
rag_pipeline = graph_builder.compile()

# --- 4. UI 헬퍼 ---
def cited_sources(answer: str, sources: list[dict]) -> list[tuple[int, dict]]:
    numbers = {int(n) for group in CITE_PATTERN.findall(answer) for n in group.split(",") if n.strip().isdigit()}
    return [(i, s) for i, s in enumerate(sources, start=1) if i in numbers]

def render_sources(answer: str, sources: list[dict]) -> None:
    used = cited_sources(answer, sources)
    if not used:
        st.caption("ⓘ 답변에 인용된 근거 문서가 없습니다.")
        return
    with st.expander(f"답변에 사용한 근거 {len(used)}건"):
        for number, source in used:
            if source.get("url", "").startswith(("https://", "http://")):
                st.markdown(f"**[근거 {number}]** [{source['type']}] {source['title']} · [원문]({source['url']})")
            else:
                st.markdown(f"**[근거 {number}]** [{source['type']}] {source['title']} · 내부 데이터셋")

# --- 5. UI 스타일 / 메뉴 ---
def change_menu(menu_key: str) -> None:
    """위젯 렌더링 전에 선택 메뉴를 갱신해 강조 표시 지연을 방지합니다."""
    st.session_state.selected_menu = menu_key


def render_sidebar_menu() -> str:
    if "selected_menu" not in st.session_state:
        st.session_state.selected_menu = "exercise"

    menu_items = [
        ("exercise", "🔍 운동 둘러보기"),
        ("qa", "💬 질문하기"),
        ("routine", "🗓️ 루틴 추천 받기"),
    ]
    active_menu = st.session_state.selected_menu

    with st.sidebar:
        st.markdown("## 🏋️‍♂️ AI 피트니스 랩")
        st.markdown("### 메뉴")

        # Streamlit은 버튼 key에 따라 st-key-menu_... 클래스를 부여합니다.
        # 모두 secondary 버튼으로 렌더링해 테마의 primary(빨강) 색상을 사용하지 않습니다.
        st.markdown(
            f"""
            <style>
            div[data-testid="stSidebar"] [class*="st-key-menu_"] button {{
                width: 100%;
                justify-content: flex-start !important;
                text-align: left !important;
                padding: 0.7rem 0.9rem !important;
                margin-bottom: 0.15rem;
                border: 1px solid transparent !important;
                border-left: 4px solid transparent !important;
                border-radius: 9px !important;
                background-color: transparent !important;
                color: inherit !important;
                box-shadow: none !important;
                font-weight: 500;
                transition: background-color 0.1s ease;
            }}
            div[data-testid="stSidebar"] [class*="st-key-menu_"] button:hover {{
                background-color: #f1f5f9 !important;
                color: inherit !important;
                border-color: transparent !important;
            }}
            div[data-testid="stSidebar"] [class*="st-key-menu_"] button:focus-visible {{
                outline: 2px solid #60a5fa !important;
                outline-offset: 2px;
            }}
            div[data-testid="stSidebar"] .st-key-menu_{active_menu} button,
            div[data-testid="stSidebar"] .st-key-menu_{active_menu} button:hover {{
                background-color: #eaf2ff !important;
                border-left: 4px solid #3b82f6 !important;
                color: #1d4ed8 !important;
                font-weight: 700 !important;
            }}
            @media (prefers-color-scheme: dark) {{
                div[data-testid="stSidebar"] [class*="st-key-menu_"] button:hover {{
                    background-color: #273449 !important;
                }}
                div[data-testid="stSidebar"] .st-key-menu_{active_menu} button,
                div[data-testid="stSidebar"] .st-key-menu_{active_menu} button:hover {{
                    background-color: #1e3555 !important;
                    color: #bfdbfe !important;
                }}
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )

        for key, label in menu_items:
            st.button(
                label,
                key=f"menu_{key}",
                use_container_width=True,
                type="secondary",
                on_click=change_menu,
                args=(key,),
            )

    return st.session_state.selected_menu

# --- 5. Streamlit 메인 화면 ---
st.set_page_config(page_title="AI 피트니스 랩", page_icon="🏋️‍♂️", layout="wide")
st.title("🏋️‍♂️ AI 피트니스 랩")
st.warning("⚠️ 본 서비스의 답변은 의학적 조언을 대신하지 않습니다. 부상 위험 시 전문가와 상담하십시오.")

df = load_csv_data()

# --- 사이드바: 페이지 이동 메뉴 ---
selected_menu = render_sidebar_menu()

# --- 메뉴 1: CSV 기반 운동 조회 ---
if selected_menu == "exercise":
    st.subheader("🔍 운동 둘러보기")
    st.caption("운동 부위와 종목을 선택하여 운동 정보와 필요한 기구를 확인하세요.")

    body_parts = sorted(df["BodyPart_KR"].dropna().unique())
    col1, col2 = st.columns(2)

    with col1:
        selected_part = st.selectbox(
            "자극 부위를 선택하세요",
            body_parts,
            index=None,
            placeholder="부위 선택",
        )

    if selected_part:
        filtered_df = df[df["BodyPart_KR"] == selected_part]
        exercises = sorted(filtered_df["Title"].unique())

        with col2:
            selected_ex = st.selectbox(
                "운동 종목",
                exercises,
                index=None,
                placeholder="운동 선택",
            )

        if selected_ex:
            ex_data = filtered_df[filtered_df["Title"] == selected_ex].iloc[0]
            st.success(f"**{selected_ex}**")
            st.caption(f"난이도: {ex_data['Level_KR']} | 필요 기구: {ex_data['Equipment_KR']}")
            description = str(ex_data["Desc"]).strip()
            if description and description.strip("*").strip():
                st.write(description)
            else:
                st.info("이 운동의 상세 설명은 등록되어 있지 않습니다.")

# --- 메뉴 2: RAG 기반 운동 질문하기 ---
elif selected_menu == "qa":
    st.subheader("💬 운동 질문하기")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if st.button("대화 지우기", key="clear_chat"):
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_sources(message["content"], message.get("sources", []))

    question = st.chat_input("예: 초보자에게 적합한 등 운동 추천해줘, 벤치프레스 주의사항 알려줘")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.status("질문을 분석하고 관련 운동 및 문서를 검색 중입니다...", expanded=False) as status:
                final_state = rag_pipeline.invoke({"question": question, "sources": [], "context": "", "answer": "", "analysis": {}})
                sources, answer = final_state["sources"], final_state["answer"]
                status.update(label=f"문서 {len(sources)}건 확보 및 분석 완료", state="complete")
                st.caption(" · ".join([f"[{s['type']}] {s['title']}" for s in sources]))

            if answer:
                st.markdown(answer)
                render_sources(answer, sources)
            else:
                st.error("답변을 생성하지 못했습니다.")

        if answer:
            st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})

# --- 메뉴 3: AI 맞춤 루틴 추천 ---
elif selected_menu == "routine":
    st.subheader("🗓️ 나만의 맞춤 운동 루틴 만들기")

    col1, col2, col3 = st.columns(3)
    with col1:
        freq = st.selectbox("주 N회 운동", ["주 2회", "주 3회", "주 4회", "주 5회"])
    with col2:
        level = st.selectbox("운동 경력", ["초보자 (0~6개월)", "중급자 (6개월~2년)", "고급자 (2년 이상)"])
    with col3:
        duration = st.selectbox("1회 운동 시간", ["30분", "60분", "90분"])

    target_part = st.text_input(
        "특별히 집중하고 싶은 부위나 목표가 있나요? (선택)",
        placeholder="예: 어깨 뽕 키우기, 코어 강화",
    )

    if st.button("🔥 맞춤 루틴 생성하기", type="primary"):
        with st.spinner("맞춤 운동 루틴을 설계 중입니다..."):
            llm = get_model()
            routine_prompt = (
                "너는 전문 피트니스 코치야. 사용자 조건에 맞춰 주간 운동 루틴(분할법)을 표 형태로 제공해.\n"
                f"조건: {freq}, {level}, {duration} / 추가목표: {target_part}\n\n"
                "운동 이름은 한국 헬스장 외래어 표기법(예: 인클라인 덤벨 프레스)을 따르고, "
                "각 운동당 추천 세트수와 반복 횟수, 그리고 간단한 팁을 적어줘."
            )
            # 다른 메뉴를 방문해도 마지막으로 생성한 루틴을 유지합니다.
            st.session_state.routine_answer = llm.invoke(routine_prompt).content

    if st.session_state.get("routine_answer"):
        st.markdown(st.session_state.routine_answer)
        st.info("💡 루틴에 포함된 운동의 상세 정보는 왼쪽 메뉴의 '운동 둘러보기'에서 확인할 수 있습니다.")
