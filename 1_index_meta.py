import os
import pandas as pd
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_chroma import Chroma
# from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings

load_dotenv()

GMS_BASE_URL = "https://gms.ssafy.io/gmsapi/api.openai.com/v1/"
API_KEY = os.environ.get("GMS_KEY")

BODYPART_MAP = {'Chest': '가슴', 'Middle Back': '등 중앙', 'Lower Back': '하배부', 'Lats': '광배근', 'Biceps': '이두근', 'Triceps': '삼두근', 'Shoulders': '어깨', 'Abdominals': '복근', 'Glutes': '둔근', 'Quadriceps': '앞벅지(대퇴사두근)', 'Hamstrings': '뒷벅지(햄스트링)', 'Calves': '종아리', 'Neck': '목', 'Traps': '승모근', 'Forearms': '전완근', 'Adductors': '내전근', 'Abductors': '외전근'}
EQUIPMENT_MAP = {'Barbell': '바벨', 'Dumbbell': '덤벨', 'Cable': '케이블', 'Machine': '머신', 'Body Only': '맨몸', 'Bands': '밴드', 'Kettlebells': '케틀벨', 'Medicine Ball': '메디신 볼', 'Exercise Ball': '짐볼', 'E-Z Curl Bar': 'EZ바', 'Foam Roll': '폼롤러', 'Other': '기타', 'None': '맨몸'}
TYPE_MAP = {'Strength': '근력', 'Plyometrics': '플라이오메트릭', 'Cardio': '유산소', 'Stretching': '스트레칭', 'Powerlifting': '파워리프팅', 'Strongman': '스트롱맨', 'Olympic Weightlifting': '역도'}
LEVEL_MAP = {'Beginner': '초급', 'Intermediate': '중급', 'Expert': '고급'}

def main():
    print("CSV 데이터를 로드하고 결측치를 제거합니다...")
    df = pd.read_csv("dataset/gym_exercises.csv").dropna()
    
    documents = []
    print(f"총 {len(df)}건의 데이터 속성을 한글화하여 색인 준비를 시작합니다.")
    
    for _, row in df.iterrows():
        # 운동명과 설명은 원문(영어) 유지! GPT가 알아서 찰떡같이 번역해줍니다.
        title = str(row['Title']).strip()
        desc = str(row['Desc']).strip()
        
        # 카테고리만 한글로 변경 (UI 필터링용)
        ex_type = TYPE_MAP.get(str(row['Type']).strip(), str(row['Type']))
        body_part = BODYPART_MAP.get(str(row['BodyPart']).strip(), str(row['BodyPart']))
        equipment = EQUIPMENT_MAP.get(str(row['Equipment']).strip(), str(row['Equipment']))
        level = LEVEL_MAP.get(str(row['Level']).strip(), str(row['Level']))
        
        text = (
            f"[운동명] {title}\n"
            f"[유형] {ex_type}\n"
            f"[자극 부위] {body_part}\n"
            f"[필요 기구] {equipment}\n"
            f"[난이도] {level}\n"
            f"[설명] {desc}"
        )
        
        # doc = Document(
        #     page_content=text,
        #     metadata={"source": "메타데이터", "title": f"{title} ({body_part})"}
        # )
        doc = Document(
            page_content=text,
            metadata={
                "source": "메타데이터",
                "title": f"{title} ({body_part})",
                "source_id": f"exercise-{int(row['Unnamed: 0'])}",
            },
        )
        documents.append(doc)
            
    print(f"변환 완료! {len(documents)}건의 메타데이터를 임베딩(Vector DB)에 저장합니다...")
    embeddings = OllamaEmbeddings(
        model="bge-m3",
        base_url="http://localhost:11434",
    )
    # embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=API_KEY, base_url=GMS_BASE_URL)
    
    # Chroma.from_documents(
    #     documents=documents,
    #     embedding=embeddings,
    #     persist_directory="./chroma_meta_db",
    #     collection_name="exercise_meta"
    # )
    
    db = Chroma(
        persist_directory="./chroma_local_app",
        embedding_function=embeddings,
        collection_name="exercise_meta_local",
    )

    batch_size = 16

    for start in range(0, len(documents), batch_size):
        batch = documents[start:start + batch_size]

        db.add_documents(
            documents=batch,
            ids=[doc.metadata["source_id"] for doc in batch],
        )

        completed = min(start + batch_size, len(documents))
        print(f"색인 진행: {completed}/{len(documents)}")

    print("메타데이터 색인이 완료되었습니다.")

if __name__ == "__main__":
    main()