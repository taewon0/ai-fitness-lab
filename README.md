# AI Fitness Lab

Kaggle에서 받은 Gym Exercise Dataset을 바탕으로 운동을 찾아보고, 운동 관련 질문에 자료 근거를 붙여 답하는 Streamlit 앱입니다. 자연어 조건 추천, 운동 정보 탐색, 맞춤 루틴 생성 화면을 제공합니다.

## 기능

- **운동 둘러보기**: 자극 부위와 운동명을 선택해 난이도, 기구, 설명을 확인합니다.
- **운동 질문하기**: 자연어 질문을 분석한 뒤 운동 메타데이터와 한국어 위키백과를 검색하고, 답변에 근거 번호와 출처를 표시합니다.
- **조건 기반 운동 추천**: 부위, 난이도, 기구 조건을 원본 CSV 값으로 필터링해 후보를 고릅니다.
- **루틴 추천**: 주간 운동 횟수, 경력, 운동 시간, 추가 목표를 바탕으로 LLM이 루틴 초안을 생성합니다. 이 기능은 검색 근거를 사용하지 않습니다.

## 실행 환경

- Python 3.11
- Ollama 서버 (`http://localhost:11434`) 및 `qwen3:8b-q4_K_M`, `bge-m3` 모델
- 네트워크 연결 (Ollama 모델 다운로드 및 한국어 위키백과 API 호출)

## 설치 및 실행

프로젝트 루트에서 PowerShell 기준으로 실행합니다.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

`WIKI_CONTACT`에는 위키백과 API 요청에 사용할 연락처를 설정합니다. 현재 앱은 로컬 Ollama를 사용하므로 `GMS_KEY`가 필요하지 않습니다. 키를 소스 코드나 공개 저장소에 올리지 마세요.

CSV는 저장소에 포함되지 않습니다. 원본 출처와 이용 조건을 확인한 뒤 `dataset/gym_exercises.csv`에 별도로 준비하세요. 자세한 형식은 [데이터 준비 안내](dataset/README.md)를 참고하세요.

Ollama를 설치하고 서버를 실행한 상태에서 다음 모델을 준비합니다.

```powershell
ollama pull qwen3:8b-q4_K_M
ollama pull bge-m3
```

최초 실행 전 운동 메타데이터를 임베딩하고 Chroma 색인을 생성합니다.

```powershell
python 1_index_meta.py
```

그다음 앱을 실행합니다.

```powershell
streamlit run app.py
```

CSV, 인덱스, `.env` 경로가 프로젝트 루트 기준 상대 경로이므로 반드시 루트에서 명령을 실행하세요. `chroma_local_app/`가 이미 존재하면 현재 색인을 재사용합니다.

> **색인 재생성 주의:** `1_index_meta.py`는 CSV 인덱스 기반 `source_id`를 사용해 문서를 추가하며, 기존 컬렉션을 비우지 않습니다. 원본 CSV를 변경하거나 전체 재색인이 필요하면 기존 DB를 먼저 백업하고 재생성하세요.

## 데이터

파일 위치: `dataset/gym_exercises.csv`

로컬에서 사용한 CSV는 2,918행, 9개 컬럼(`Title`, `Desc`, `Type`, `BodyPart`, `Equipment`, `Level`, `Rating`, `RatingDesc` 및 인덱스 컬럼)입니다. 결측치는 설명 1,550건, 기구 32건, 평점 1,887건, 평점 설명 2,056건입니다.

현재 앱과 색인 스크립트는 `dropna()`로 행의 모든 컬럼을 검사합니다. 따라서 실제 색인·화면에서 사용할 수 있는 데이터는 501건입니다. 평점처럼 앱에서 사용하지 않는 컬럼의 결측도 행을 제외하는 원인이 됩니다.

데이터 출처는 사용자가 제공한 Kaggle의 **Gym Exercise Dataset**입니다. 정확한 Kaggle 데이터셋 URL, 작성자, 라이선스는 프로젝트 파일에서 확인되지 않았습니다. 외부 제출 또는 재배포 전 원본 페이지와 라이선스, 출처 표기 조건을 확인해 기록하세요.

## 구성

```text
.
├── app.py                          # Streamlit UI, 검색/응답 그래프
├── 1_index_meta.py                 # CSV 메타데이터 임베딩 및 Chroma 색인
├── dataset/gym_exercises.csv       # 별도 준비하는 로컬 운동 데이터 (Git 제외)
├── chroma_local_app/               # 생성된 로컬 Chroma 데이터베이스 (Git 제외)
├── .env.example                    # 환경변수 템플릿
├── requirements.txt                # 실행에 필요한 직접 의존성
└── docs/
    ├── rag-pipeline.md             # 검색·응답 파이프라인 설계
    ├── requirements-traceability.md # PDF 요구사항과 구현 현황
    └── verification-checklist.md   # 실행 검증 체크리스트
```

## 설정

| 변수 | 용도 |
| --- | --- |
| `GMS_KEY` | 기존 GMS 연동용 키. 현재 로컬 Ollama 코드에서는 사용하지 않습니다. |
| `WIKI_CONTACT` | 위키백과 API User-Agent에 넣을 연락처입니다. |

현재 앱과 색인 스크립트는 로컬 Ollama의 `qwen3:8b-q4_K_M`와 `bge-m3`를 사용합니다. 앱의 Chroma 경로는 `chroma_local_app/`이며, 평가 스크립트의 DB(`eval/db/`)와 별개입니다.

## 알려진 제한 및 요구사항 현황

요구사항 PDF의 기능 항목 F2001~F2016과 비기능 항목 NF2001~NF2007을 검토했습니다. 데이터 적재, 기본 검색·답변, 인용, 운동 탐색은 코드에서 확인되지만, MMR 검색, RAG 사용 전후 품질 비교, 중복 없는 재색인 등은 현재 구현에서 확인되지 않았습니다. 세부 상태와 검증 필요 사항은 [요구사항 추적표](docs/requirements-traceability.md)를 참고하세요.

외부 위키 요청에 타임아웃이 설정되어 있지 않으며, 위키 검색 문서는 질문마다 임시 벡터 저장소를 만들어 임베딩합니다. 루틴 생성은 검색 근거에 기반하지 않습니다. 앱은 의료 진단이나 치료를 제공하지 않으며, 부상 또는 건강 문제가 있으면 전문가와 상담하세요.

## 문서

- [RAG 파이프라인 설계](docs/rag-pipeline.md)
- [요구사항 추적표](docs/requirements-traceability.md)
- [검증 체크리스트](docs/verification-checklist.md)

## 보안 및 배포

- `.env`에는 비밀 키만 저장하고 공개 저장소에 포함하지 마세요.
- `chroma_meta_db/`, `chroma_local_app/`, `eval/db/`는 재생성 가능한 로컬 인덱스이며 Git에서 제외합니다.
- 요구사항 PDF에는 대외비 표시가 있으므로 허가 없이 외부에 배포하지 마세요.
- CSV 데이터와 위키백과 문서의 저작권·라이선스 및 출처 표기 조건을 확인한 뒤 결과물을 배포하세요.

## 저장소 포함 범위

- 앱·색인 코드, 평가 스크립트, 문서, 빈 `.env.example`, 채점 기준과 RAGAS 의존성 제약 파일을 포함합니다.
- `.env`, Streamlit 비밀 설정, 가상환경, 캐시, DB, 평가 실행 결과·로그·코드 보존본·환경 진단 파일은 제외합니다.
- 운동 CSV와 원문 인용이 포함된 평가 질문 세트(`eval/dev.jsonl`, `eval/test.jsonl`, `eval/smoke.jsonl`)는 출처·재배포 조건이 정리될 때까지 로컬에 보관합니다.
- 내부 요구사항 PDF와 로컬 작업 인계 문서는 제외합니다. 기존 로컬 파일은 삭제하지 않습니다.
- 평가 가이드에 나오는 기존 실행 결과와 고정 계획은 로컬 자료입니다. 공개 저장소만으로 기존 평가 결과를 재분석할 수는 없으며, 평가 스크립트를 사용할 때는 질문 세트와 필요한 계획·판정 자료를 별도로 준비해야 합니다.
