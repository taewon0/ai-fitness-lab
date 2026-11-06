# AI Fitness Lab 로컬 LLM 전환 및 성능 평가 가이드

최신 실행 상태: final-001(1번)은 360/360건, 오류·출력 잘림 0건으로 완료됐습니다. 다음 채팅에서는 인계 문서와 eval/final-progress.json을 확인하고 final-002·003만 진행하세요. 완료된 1번은 재실행하지 마세요.


다른 채팅에서 재개할 때는 [최신 인계 문서](local-rag-evaluation-handoff.md)를 먼저 확인하세요. 최종 명령과 상태·채점 기준이 정리되어 있습니다.

작성 기준: 2026-10-28. 프로젝트 루트: C:\tw\streamlit_lab.
이 문서와 scripts/local_rag_eval.py는 가이드용으로 추가했다.
기존 app.py, 1_index_meta.py, 원본 CSV, 기존 chroma_meta_db는 수정하지 않았다.
모델 설치·다운로드·실제 추론·성능 측정은 아직 수행하지 않았다.

## 1. 이 PC에서 가능한가?

가능하다. nvidia-smi로 확인한 실제 GPU는 NVIDIA GeForce RTX 5060 Ti이며
VRAM은 16,311 MiB(16GB 제품), 드라이버는 596.49이다.
RAM 32GB, i7 11세대, 저장소 1TB는 사용자가 제공한 사양이다.
이번에는 GPU만 직접 확인했다.

출발 구성:
- 생성/질의 분석: qwen3:8b-q4_K_M
- 임베딩: bge-m3
- 실행: Windows용 Ollama
- 앱: 기존 Streamlit + LangChain + LangGraph + Chroma
- 컨텍스트: 8,192토큰, 출력 상한: 1,024토큰, temperature=0, think=false
- 평가: 한 번에 한 요청, 로컬 HTTP API 호출, 수동 정답 채점

Qwen3 8B Q4_K_M의 다운로드 파일은 약 5.2GB이며, BGE-M3는 약 1.2GB이다.
파일 크기와 실행 VRAM은 다르다. KV 캐시·컨텍스트·동시 실행 모델에 추가 VRAM이 필요하다.
이 구성은 이 PC에서 시도할 합리적인 시작점이라는 판단이다.
실제 GPU 적재와 속도는 설치 후 ollama ps와 실측으로 확인한다.
처음부터 30B 이상 모델이나 파인튜닝을 진행할 필요는 없다.

근거:
- [Ollama Qwen3 8B Q4_K_M](https://ollama.com/library/qwen3:8b-q4_K_M)
- [Ollama BGE-M3](https://ollama.com/library/bge-m3)
- [Windows 지원](https://docs.ollama.com/windows)

## 2. 어떤 결과를 만들 것인가?

현재 전처리와 501건을 유지하며, 두 종류의 실험을 분리한다.

| 실험 | 고정하는 것 | 바꾸는 것 | 알 수 있는 것 |
| --- | --- | --- | --- |
| RAG 효과 | 모델·질문·현재 501문서 코퍼스·출력 정책 | 자료 없음 / 검색 자료 제공 | 로컬 모델의 RAG 전후 답변 품질 |
| 검색·속도 개선 | 모델·질문·현재 501문서 코퍼스·출력 정책 | 검색 방식·문서 수·캐시 등 한 가지 | 개선 항목별 품질·지연 변화 |

로컬 전환 전 GMS와 로컬 전환 후 결과를 직접 비교해 “RAG로 개선”했다고 해석하지 않는다.
모델과 실행 환경이 달라 원인을 분리할 수 없기 때문이다.
실험명에는 “프로젝트 종료 후 로컬 환경에서 수행한 후속 평가”라고 기록한다.

RAG가 정확도를 높이더라도 응답시간은 늘어날 수 있다.
RAG 품질 실험과 속도 최적화 실험은 서로 다른 질문에 답한다.
데이터 확대나 전처리 변경은 이번 계획에 포함하지 않는다.
향상되지 않은 지표도 기록하고, 유리한 결과만 골라 전체 성능으로 표현하지 않는다.

## 3. Ollama 설치와 모델 준비

[Windows 설치 페이지](https://ollama.com/download/windows)에서 Ollama를 설치한다.
설치 후 새 PowerShell 창을 연다. Windows 앱이 백그라운드에서 서버를 실행하므로
이미 실행 중이면 별도로 ollama serve를 실행할 필요가 없다.

프로젝트 루트에서:

```powershell
Set-Location C:\tw\streamlit_lab
ollama --version
ollama pull qwen3:8b-q4_K_M
ollama pull bge-m3
ollama list
Invoke-RestMethod -Uri http://localhost:11434/api/tags
```

로컬 모델 사용에는 GMS_KEY나 Ollama 클라우드 키가 필요 없다.
모델을 처음 다운로드할 때는 인터넷이 필요하다.
초기 다운로드 후 CSV 기반 실험은 외부 모델 API 없이 실행한다.
위키백과 실시간 수집을 계속 사용하면 그 부분은 인터넷이 필요하다.

짧은 응답 확인:

```powershell
ollama run qwen3:8b-q4_K_M
```

질문을 입력하고 종료할 때 /bye를 입력한다.
thinking 비활성화 등 정확한 평가 설정은 아래 Python 도구가 지정한다.

다른 PowerShell에서 GPU 사용 확인:

```powershell
ollama ps
nvidia-smi
```

ollama ps의 PROCESSOR가 100% GPU인지 확인한다.
CPU/GPU 혼합이면 속도를 단정하지 말고 모델 크기·컨텍스트·다른 GPU 프로그램을 점검한다.
로컬 클라이언트 사용에는 CUDA Toolkit이나 PyTorch를 별도로 설치할 필요가 없다.
Ollama가 추론 런타임을 제공한다.

공식 안내:
- [로컬 인증](https://docs.ollama.com/api/authentication)
- [GPU 적재 확인·keep_alive·컨텍스트](https://docs.ollama.com/faq)

## 4. Python 환경 복구

현재 .venv\Scripts\python.exe는 실행되지 않았다.
설치되어 있지 않은 C:\Users\SSAFY\AppData\Local\Programs\Python\Python311\python.exe를 참조한다.
기존 가상환경을 삭제하지 말고 새 .venv-local을 만든다.

이 프로젝트 README 기준은 Python 3.11이다.
먼저 Python 설치 상태를 확인한다.

```powershell
py --list
```

Python 3.11이 없다면 공식 [Python Windows 배포 페이지](https://www.python.org/downloads/windows/)에서
프로젝트에 사용할 Python을 설치한다. 아래 명령은 3.11이 설치된 경우 기준이다.
다른 지원 버전을 선택하면 py -3.11 부분도 해당 버전으로 맞추고 의존성 설치를 확인한다.

```powershell
py -3.11 -m venv .venv-local
.\.venv-local\Scripts\python.exe -m pip install --upgrade pip
.\.venv-local\Scripts\python.exe -m pip install streamlit pandas python-dotenv requests langchain-chroma langchain-text-splitters langgraph langchain-ollama
.\.venv-local\Scripts\python.exe -m pip check
.\.venv-local\Scripts\python.exe -m pip freeze | Set-Content -LiteralPath requirements-local.lock.txt -Encoding UTF8
```

가상환경 활성화 없이 실행 파일을 직접 지정하므로 PowerShell 실행 정책 변경이 필요 없다.
이 설치는 원래 requirements.txt를 바꾸지 않는다.
설치가 성공한 버전을 lock 파일로 남기고, 실험 도중 패키지를 업데이트하지 않는다.

## 5. 기존 앱을 로컬 모델에 연결

다음은 사용자가 app.py와 1_index_meta.py에 적용할 변경 가이드다.
이번 가이드 작성 작업에서는 기존 앱 코드를 수정하지 않았다.

### 5.1 app.py의 모델과 임베딩

기존 import를 교체한다.

```python
from langchain_ollama import ChatOllama, OllamaEmbeddings
```

ChatOpenAI, OpenAIEmbeddings import를 제거하고 다음 팩터리를 사용한다.

```python
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
```

이렇게 get_model을 교체하면 질의 분석·답변 생성·루틴 생성 호출이 함께 바뀐다.
GMS_KEY 확인 때문에 중단하는 기존 get_model 코드는 남기지 않는다.
GMS_BASE_URL은 더 이상 쓰지 않으면 제거한다.
.env를 쓰는 다른 설정(WIKI_CONTACT 등)은 유지할 수 있다.

관련 공식 문서:
- [ChatOllama](https://docs.langchain.com/oss/python/integrations/chat/ollama)
- [OllamaEmbeddings](https://docs.langchain.com/oss/python/integrations/embeddings/ollama)

### 5.2 위키 문서 임베딩도 변경

app.py의 Chroma.from_texts 내부에 남아 있는 OpenAIEmbeddings도 교체한다.

```python
wiki_db = Chroma.from_texts(
    texts=chunks,
    embedding=get_embeddings(),
)
```

모델 팩터리만 바꾸고 이 부분을 남기면 지식 질문에서 여전히 GMS 호출이 발생한다.
위키를 질문마다 다시 임베딩하는 구조는 우선 유지하고, 캐시는 별도 실험으로 추가한다.
HTTP 수집에는 timeout=(5, 20) 등 연결·읽기 제한을 두고 실패를 구분해 기록한다.

### 5.3 1_index_meta.py의 임베딩과 저장 경로

다음 import와 객체를 사용한다.

```python
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(
    model="bge-m3",
    base_url="http://localhost:11434",
)

# 기존 documents 구성은 첫 전환 시 그대로 유지한다.
# 각 행을 Document로 만드는 루프에서 source_id를 함께 저장하면 추적하기 쉽다.
# doc.metadata["source_id"] = f"exercise-{int(row['Unnamed: 0'])}"

Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="./chroma_local_app",
    collection_name="exercise_meta_local",
    ids=[doc.metadata["source_id"] for doc in documents],
)
```

source_id 대입을 루프 안에 실제로 추가한 다음 ids를 사용해야 한다.
source_id 없이 기존 문서를 그대로 사용한다면 위 ids 줄은 바로 실행할 수 없다.

기존 chroma_meta_db를 새 임베딩 검색에 사용하지 않는다.
벡터 길이가 같아도 모델이 다르면 같은 의미 공간이 아니다.
색인과 검색 양쪽에 bge-m3를 사용한다.
임베딩 모델을 변경하면 기존 501건을 동일하게 유지한 채 별도 경로에 재색인하고 모델 버전을 기록한다.

현재 전처리 정책과 501건의 데이터를 모든 비교 실험에서 고정한다.
이 프로젝트는 운동 수를 늘리는 대신, 현재 제공하는 운동 정보의 정확성과 답변의 근거 충실성을 우선한다.
평가 질문도 이 데이터 범위를 기준으로 구성하고, 근거가 없는 내용은 한계를 명확히 안내하는지 평가한다.

### 5.4 실행 및 남은 GMS 호출 점검

```powershell
.\.venv-local\Scripts\python.exe 1_index_meta.py
.\.venv-local\Scripts\python.exe -m streamlit run app.py
rg -n "ChatOpenAI|OpenAIEmbeddings|GMS_KEY|GMS_BASE_URL" app.py 1_index_meta.py
```

실제 코드 변경 후 실행하는 명령이다. 인덱스 생성 경로와 앱에서 여는 경로가 같아야 한다.
상수나 주석이 검색되어도 실제 GMS 호출인지 구분해 확인한다.
변경 후 Streamlit 프로세스를 재시작해 이전 캐시 객체를 다시 사용하지 않도록 한다.

질문 분석은 현재 프롬프트 JSON을 수동 파싱한다.
로컬 모델 전환 후 JSON 파싱 성공률을 기록하고, 실패하면 규칙 기반 처리로 돌아가는지 확인한다.
네트워크 예외 전체가 현재 fallback 대상은 아니므로 연결 실패도 별도로 처리해야 한다.
Adductors/Abductors는 프롬프트 허용값과 app의 허용 집합이 일치하지 않는 점도 평가 대상이다.

## 6. 준비된 별도 평가 도구 사용

scripts/local_rag_eval.py는 Streamlit 앱과 독립적으로 실행한다.
app.py를 import하지 않으므로 화면 실행·세션 상태에 의존하지 않는다.

평가 범위는 다음으로 제한한다.
- 고정 CSV의 운동 지식 질의응답
- 같은 모델의 no_rag / 벡터 검색 rag / MMR 검색 rag_mmr 비교
- 한국어 원질의를 BGE-M3에 직접 임베딩
- 질의 분석 LLM, 영문 검색어 변환, 실시간 위키 수집, 루틴, UI는 제외

따라서 이 도구의 total_s는 CSV 검색·답변 생성 구간의 시간이다.
“기존 앱 전체 응답시간”으로 보고하지 않는다.
기존 앱의 영문 검색어 경로와 문서 구성도 완전히 같지는 않으므로
“로컬 CSV-RAG 평가 기준선”이라는 실험명을 사용한다.
전체 서비스 평가는 11절에서 별도로 측정한다.

### 6.1 모델 API 없이 데이터 수 점검

```powershell
.\.venv-local\Scripts\python.exe scripts\local_rag_eval.py prepare --corpus baseline --dry-run
```

현재 CSV에서 예상되는 값:

| 코퍼스 | 행/문서 수 | 고유 운동명 수 |
| --- | ---: | ---: |
| baseline | 501 | 495 |

baseline은 현재 전처리를 그대로 적용한 고정 데이터라는 뜻이다.
이 가이드의 모든 명령은 --corpus baseline을 사용한다.
원본 CSV와 결측 제거·중복 처리 정책을 변경하지 않는다.
501문서와 495개 고유 운동명이라는 차이는 현재 데이터의 특성이며, 이번 평가에서 중복 제거를 추가하지 않는다.
실험마다 CSV 해시와 문서 수를 확인해 같은 데이터가 사용됐는지 기록한다.
자료가 충분하다는 판단과 개별 정보의 정확성 검증은 구분하고, 답변은 실제 근거와 대조해 채점한다.

### 6.2 로컬 평가 DB 생성

Ollama와 bge-m3가 준비된 다음 실행한다.

```powershell
.\.venv-local\Scripts\python.exe scripts\local_rag_eval.py prepare --corpus baseline
```

생성 위치는 eval/db/baseline이다.
앱용 chroma_local_app이나 기존 chroma_meta_db를 변경하지 않는다.
기존 경로가 있으면 덮어쓰지 않고 중단한다.
실패한 색인을 다시 만들 때는 해당 실험 폴더만 다른 이름으로 보관한 뒤 새로 만든다.
성공한 DB의 manifest.json에는 CSV 해시와 모델 정보가 저장된다.

### 6.3 3개 질문으로 연결 확인

```powershell
.\.venv-local\Scripts\python.exe scripts\local_rag_eval.py seed
.\.venv-local\Scripts\python.exe scripts\local_rag_eval.py run --dataset eval/smoke.jsonl --corpus baseline --methods no_rag rag --repeats 1 --run-id smoke-001
```

3문항은 실측 성능을 주장하기 위한 표본이 아니다.
API 연결, 검색, 한국어 출력, 파일 저장을 확인하는 용도다.
각 문항의 reference_answer는 CSV 사실을 바탕으로 작성했지만 최종 평가셋은 사람이 다시 검토한다.

각 실행은 다음을 생성한다.

| 파일 | 내용 |
| --- | --- |
| config.json | 모델·옵션·코퍼스 해시·질문 해시·스크립트 해시·계획 요청 수 |
| answers.jsonl | 질문, 답변, 검색 문서, 방법, 반복 번호, 시간, 토큰, 오류 |
| review.csv | 방법 이름을 숨긴 수동 채점표 |
| summary.json | 채점 후 summarize로 생성하는 지표 |

run-id가 같은 폴더는 덮어쓰지 않는다.
재실행 시 smoke-002처럼 새 이름을 사용한다.
중단된 실행은 계획 요청 수가 맞지 않으면 완료 실험으로 요약하지 않는다.

## 7. 정식 평가셋과 에이전트 검토

2026-10-28: eval/dev.jsonl(20문항), eval/test.jsonl(60문항), eval/grading-rubric.md를 준비했다.
평가 질문 작성, 원문 대조, review.csv 채점, 결과 집계는 사용자의 요청에 따라 에이전트가 담당한다.
아래의 수동 검토·채점 안내는 에이전트가 수행하는 절차이며 사용자가 직접 채점할 필요는 없다.
결과는 에이전트의 근거 기반 평가라고 표시하며, 독립적인 전문가·사람 검증을 수행했다고 표현하지 않는다.
실행 후 eval/runs/<run-id> 경로를 알려주면 에이전트가 이어서 검토한다.
기존 전처리와 501문서는 그대로 유지한다.

새 개발셋의 첫 실행(Git Bash, 활성화된 .venv):

```bash
python scripts/local_rag_eval.py run --dataset eval/dev.jsonl --corpus baseline --methods no_rag rag --repeats 1 --run-id dev-001
```

20문항 × 2방법 = 40요청이다. 우선 이 실행을 마친 뒤 에이전트에게 검토를 요청한다.
최종 test.jsonl은 개발셋 검토와 실험 설정 확정 후 실행한다.

원래의 평가셋 설계 원칙은 아래와 같다.

권장 출발 규모는 개발용 20~40문항, 최종 테스트용 50~60문항이다.
표본 수는 시간에 맞춰 정하되 실험 전에 고정한다.
예: test 60문항 중 설명 보유 질문 45개, 설명 부족·답변 불가 질문 15개.
같은 운동의 단순한 바꿔 말하기를 대량으로 늘려 독립적인 문항 수처럼 계산하지 않는다.
동일 운동·유사 질문은 가능한 한 같은 split에 묶는다.

eval/dev.jsonl, eval/test.jsonl을 UTF-8 JSONL로 작성한다.
한 줄이 한 JSON 객체다. seed의 형식을 따라 작성한다.

```json
{"id":"k001","kind":"knowledge","split":"test","question":"클로즈 그립 벤치프레스의 손 간격과 강조되는 근육은?","relevant_ids":["exercise-927"],"expected_abstain":false,"reference_answer":"일반 벤치프레스보다 손 간격이 좁고 삼두근을 더 강조합니다.","required_facts":["더 좁은 손 간격","삼두근 강조"]}
```

정답 근거는 eval/db/baseline/corpus.json에서 확인할 수 있다.
source_id는 원본 CSV의 Unnamed: 0 값에서 만든 exercise-번호다.

질문 유형:
- 특정 운동의 자극 근육·기구·난이도
- CSV 설명에 실제로 존재하는 동작 특성
- 서로 다른 운동의 비교: 필요한 자료 ID를 모두 기재
- 영문 운동명의 한국어 표현과 별칭
- CSV에 없는 수치·개인 정보·근거 부족 질문

질문과 정답은 검색 결과를 보며 계속 고치지 않는다.
개발셋에서 개선하고, 마지막에 고정 테스트셋으로 평가한다.
정답이 코퍼스에 없는 문항은 expected_abstain=true로 표시한다.
의학적 정답이나 효과를 CSV만으로 확정하지 않는다.
주관적인 “최고의 운동”을 단일 정답으로 채점하지 않는다.

동일한 사실을 여러 문서가 뒷받침하면 relevant_ids에 허용 ID를 충분히 기록한다.
정답 목록이 불완전하면 검색 Recall이 실제보다 낮아질 수 있다.
비교 질문에서는 Hit@5만으로 충분하지 않으며 필요한 근거가 모두 검색됐는지도 확인한다.

모든 비교 조건에서 같은 501문서 코퍼스를 사용한다.
코퍼스에 답이 있는 문항은 사실 정확성과 검색 품질을 평가하고,
답이 없는 문항은 정보를 지어내지 않고 적절히 한계를 밝히는지 평가한다.
답변할 수 없는 문항의 해결을 위해 이번 실험에서 운동 데이터를 추가하지 않는다.

답변 생성과 동일한 LLM이 만든 정답을 검토 없이 그대로 정답으로 사용하지 않는다.

## 8. 실험 실행과 에이전트 채점

### 8.1 RAG 전후

```powershell
.\.venv-local\Scripts\python.exe scripts\local_rag_eval.py run --dataset eval/test.jsonl --corpus baseline --methods no_rag rag --repeats 3 --run-id rag-ab-001
```

질문 60개 × 방법 2개 × 반복 3회 = 총 360회 생성 요청이다.
시간이 많이 들 수 있으므로 먼저 개발셋 일부와 repeats 1로 확인한다.
구체적인 실행시간은 PC에서 첫 실측을 한 뒤 산정한다.

모델·임베딩 warm-up은 측정 전에 수행한다.
방법과 질문 순서는 고정 난수로 섞어 순서 편향을 줄인다.
temperature=0도 GPU 실행의 비결정성을 완전히 없애지는 않으므로 반복한다.
반복 3회는 질문 60개를 180개의 독립 질문으로 늘리는 것이 아니다.

### 8.2 채점

에이전트가 review.csv에서 correct, unsupported, abstained를 각각 0/1로 입력한다.
엄격한 근거 기준과 애매한 사례 처리는 eval/grading-rubric.md를 따른다.
성공·출력 잘림 응답은 모두 채점하며 미채점이 있으면 요약을 중단한다.

| 컬럼 | 1의 의미 |
| --- | --- |
| correct | 필수 사실을 충족하고 중대한 사실 오류가 없음. 답변 불가 문항은 적절히 모른다고 답함 |
| unsupported | 답변에 고정 근거·검증된 정답 자료로 확인되지 않는 사실 주장이 있음 |
| abstained | 알 수 없거나 자료가 부족하다고 명확하게 답함 |

unsupported는 “인용 표시가 없다”라는 뜻이 아니다.
자료를 받지 않은 no_rag 답변도 정답 자료를 기준으로 같은 기준으로 채점한다.
retrieved context만으로 no_rag를 자동 오답 처리하지 않는다.
내용이 맞는지와 인용이 올바른지는 별도 기준이다.
unsupported=1인 응답이 항상 사실상 거짓이라는 뜻도 아니다. 이 지표는 근거 확인 가능성을 나타낸다.

자유 서술식 사실 채점 기준은 실험 전에 정한다.
애매하면 notes에 판단 이유를 적고 일부를 두 번째 사람과 교차 확인한다.
review.csv의 result_id와 행을 삭제하거나 임의로 바꾸지 않는다.
방법 이름은 숨겨져 있지만 인용 형태로 RAG 여부를 추정할 수 있으므로 완전한 이중맹검이라고 부르지 않는다.
CSV 편집 후 UTF-8로 저장하고 긴 answer/evidence 셀이 손상되지 않았는지 확인한다.
answers.jsonl은 원본 로그이므로 별도로 유지한다.

### 8.3 결과 집계

```powershell
.\.venv-local\Scripts\python.exe scripts\local_rag_eval.py summarize --run-id rag-ab-001
```

API 오류는 전체 정확도에서 실패로 포함한다.
응답시간은 성공 요청과 전체 요청의 통계를 구분한다.
출력이 상한에서 잘린 비율도 함께 보고한다.
잘린 응답을 조용히 제외해 정확도나 시간을 유리하게 만들지 않는다.
상한을 늘렸다면 모든 비교 조건을 새 설정으로 다시 실행한다.

## 9. 지표의 뜻과 향상률 계산

| 지표 | 계산 또는 의미 |
| --- | --- |
| accuracy_all_attempts | 정답 판정 수 / 전체 생성 시도 수. API 실패 포함 |
| Hit@k | 정답 ID 중 하나라도 상위 k에 들어간 질문 비율 |
| Recall@k | 필수 근거 그룹 중 하나 이상의 대체 문서를 검색한 그룹의 비율을 문항별로 평균 |
| unsupported_answer_rate_successful | 근거를 확인할 수 없는 주장이 하나 이상 있는 성공 응답 비율 |
| unanswerable_correct_rate | 답변 불가 문항에서 올바르게 응답한 비율 |
| error_rate / truncation_rate | API 실패율 / 출력 상한으로 잘린 비율 |
| median_total_s_successful | 성공 요청의 검색·생성 총시간 중앙값 |
| p95_total_s_successful | 성공 요청의 검색·생성 총시간 95백분위 |
| median_embedding_s | 질문 임베딩 시간 중앙값 |
| median_retrieval_s | 벡터가 준비된 뒤 DB 검색 시간 중앙값 |
| median_generation_s | 생성 HTTP 요청부터 전체 답변 수신까지 중앙값 |

Hit/Recall은 relevant_ids가 있는 answerable 문항만 계산한다.
새 평가셋은 relevant_groups로 같은 사실을 뒷받침하는 대체 문서를 묶는다.
한 그룹에서 하나만 검색해도 해당 근거를 확보한 것으로 보며, 비교 질문은 양쪽 그룹을 모두 확보해야 Recall=1이다.
relevant_groups가 없는 과거 평가셋은 기존처럼 각 ID를 별도의 필수 근거로 취급한다.
no_rag에는 검색 지표를 부여하지 않는다.
BGE-M3 OllamaEmbeddings 경로는 여기서 밀집 벡터 검색만 사용한다.
BGE-M3의 sparse/multi-vector 기능까지 자동으로 사용하는 것은 아니다.

아래 숫자는 계산 예시이며 실제 측정 결과가 아니다.

- 정확도 60% → 80%: +20%p, 상대 향상률 (0.80-0.60)/0.60×100 = 33.3%
- 응답시간 12초 → 8초: 감소율 (12-8)/12×100 = 33.3%
- 검색 실패율 20% → 10%: -10%p, 상대 감소율 50%
- 기준값이 0이면 상대 향상률을 계산하지 말고 절대 차이를 보고한다.

60개 질문을 3회 반복한 결과에는 “60개 고정 질문, 각 3회 반복”이라고 쓴다.
정확도의 불확실성을 추정하려면 질문 단위의 paired bootstrap 등을 사용한다.
각 반복을 독립 표본으로 가정해 신뢰구간을 과도하게 좁히지 않는다.
소규모 평가 결과를 모든 운동 질문에 대한 일반 성능으로 확대 해석하지 않는다.

Ollama의 eval_count/eval_duration은 생성 토큰 처리율을 계산하는 데 쓴다.
이 값은 검색·프롬프트 처리·모델 로딩이 포함된 사용자 대기시간과 다르다.
스크립트는 서버 시간의 nanosecond를 초로 변환해 기록한다.
stream=false이므로 첫 토큰 지연(TTFT)은 측정하지 않는다.
TTFT가 필요하면 스트리밍의 첫 content 수신 시각을 별도로 기록해야 한다.
모델 로딩시간도 load_s로 남으므로 중간 모델 재적재 여부를 확인한다.

공식 API 필드: [Ollama chat API](https://docs.ollama.com/api/chat)

## 10. 개선 실험을 순서대로 수행

### A. 검색 다양성: 유사도와 MMR

```powershell
.\.venv-local\Scripts\python.exe scripts\local_rag_eval.py run --dataset eval/dev.jsonl --corpus baseline --methods rag rag_mmr --repeats 1 --run-id mmr-dev-001
```

준비된 MMR 설정은 k=5, fetch_k=20, lambda_mult=0.5다.
같은 코퍼스에서 답변 정확도·Recall·중복 문서 노출·검색시간을 비교한다.
MMR이 항상 더 정확하거나 빠르다는 전제는 두지 않는다.
개발셋에서 설정을 고른 뒤 고정 테스트셋으로 3회 반복한다.

### B. 검색 문서 수

개발셋에서 --k 3과 --k 5 등을 비교한다.
각 run-id를 다르게 하고 변경한 설정을 기록한다.
문서 수를 줄이면 입력 토큰과 시간이 줄 수 있지만 필요한 근거가 빠질 수도 있다.
정확도·Recall·출력 토큰·응답시간을 함께 본다.
각 k 설정의 반복 실행으로 최적화하되 테스트셋을 계속 보면서 고르지 않는다.

### C. 캐시와 사전 색인

현재 별도 평가 도구는 운동 문서의 임베딩을 미리 저장한다.
실시간 위키 수집/재임베딩의 비용은 이 도구에 포함되지 않는다.
따라서 이 결과만으로 “위키 캐시로 전체 앱이 빨라졌다”고 주장할 수 없다.

앱 속도 개선 실험은 다음 순서로 구현한다.
1. 문서 URL·수집시각·본문 해시를 가진 위키 스냅샷을 저장한다.
2. 같은 스냅샷으로 “요청마다 문서 임베딩”과 “기존 임베딩 재사용”을 비교한다.
3. 질문 임베딩 캐시는 별도 조건으로 추가한다.
4. 첫 질문(cache miss)과 같은 질문 반복(cache hit)을 구분한다.
5. 완성된 답변 자체를 캐시한 결과는 “답변 캐시 적중 지연”이라고 별도 보고한다.

같은 문서를 반복 수집할 때 실제 외부 HTTP 시간이 달라지는 실험은 별도로 보고한다.
고정 스냅샷은 네트워크를 제외한 임베딩·검색 비교이며, 라이브 HTTP 전체 지연과 다르다.

## 11. 앱 전체 평가와 조건 추천 평가

별도 도구로 검색·생성의 효과를 확인한 다음 앱을 측정한다.

앱 전체 요청에서 perf_counter로 다음 구간을 저장한다.
- question_analysis_s: LLM JSON 분석과 규칙 보강
- csv_filter_s: 추천 후보 필터
- exercise_search_s: 운동 DB 질문 임베딩+검색
- wiki_fetch_s: 검색과 본문 HTTP 수집
- wiki_index_s: 청킹+임베딩+색인
- wiki_search_s: 위키 관련 청크 검색
- generation_s: 최종 답변 생성
- total_pipeline_s: rag_pipeline.invoke 전체
- UI까지의 대기시간: 필요하면 별도 브라우저 측정

```python
from time import perf_counter

start = perf_counter()
final_state = rag_pipeline.invoke(initial_state)
pipeline_seconds = perf_counter() - start
# 질문 ID, 모델/데이터 버전, 단계별 시간, 오류 상태와 함께 기록
```

기존 app.py를 평가 스크립트에서 import하면 Streamlit 최상위 코드가 함께 실행된다.
반복 평가가 필요하면 검색·생성 로직을 서비스 모듈로 분리하고 UI와 평가 도구가 공유한다.
이번에 추가한 평가 도구는 이 리팩터링을 수행하지 않았다.

조건 추천은 “CSV의 명시 조건 일치”를 평가한다.
정답 질문에는 expected body_parts/level/equipment를 기록한다.
- 의도 분류 정확도와 조건별 추출 정확도
- 최종 추천 목록의 조건 위반 비율
- 충분한 후보가 있는 질문에서 결과 없음 비율
- 실제 후보가 없는 질문의 올바른 안내 비율
- 존재하지 않는 운동 생성 비율
- 결과의 부위 다양성과 반복 노출

필터 함수 자체가 조건을 만족하도록 작성돼 있다면 필터만 평가해 높은 점수를 얻을 수 있다.
자소서의 “추천 정확도”는 사용자 질문 해석부터 최종 응답까지 어느 범위인지 명시한다.
여러 부위가 지정되면 OR/AND 정책과 필요한 결과 구성을 먼저 정한다.
루틴의 세트·반복·회복시간은 현재 CSV만으로 정답을 검증할 수 없으므로 별도 자료가 필요하다.

## 12. 실험 조건과 증빙 보관

실험마다 보관:
- GPU/VRAM, RAM, CPU, Windows, 드라이버
- Python 버전, requirements-local.lock.txt
- ollama --version, ollama list, ollama ps 출력
- 모델 tag와 digest, 임베딩 모델 정보
- CSV·질문·코드 해시, 현재 전처리 정책이 유지됐다는 기록
- temperature, seed, think, num_ctx, num_predict, k, 반복 횟수
- 질문·정답·검색 문서·답변·사람 채점·오류·원시 시간 로그

예시:

```powershell
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
ollama --version
ollama list
ollama ps
.\.venv-local\Scripts\python.exe --version
```

원시 로그와 채점표는 유지한다. API 오류를 삭제하지 않는다.
비교 실험 중 게임·다른 GPU 모델·큰 다운로드 등 부하 변화를 피하고 실행 조건을 기록한다.
temperature=0, 모델 파일 고정, 질문 고정으로 비교 가능성을 높인다.
prefix/KV cache도 속도에 영향을 줄 수 있으므로 방법 순서를 섞고 cold/warm 조건을 명시한다.
현재 도구는 warm 모델의 순차 요청 실험이며, 캐시가 전혀 없는 통제 실험은 아니다.

## 13. 자소서에 쓸 때

실측 전에는:
“외부 API 종료 후 로컬 모델 기반 평가 환경을 설계하고 비교 지표를 정의했다.”

실측 후에는 다음 빈칸을 실제 값으로 채운다.
“RTX 5060 Ti 16GB 환경에서 동일한 로컬 모델과 고정 질문 N개를 사용해
RAG 적용 전후를 비교했다. 문항별 기준에 따른 정답률이 A%에서 B%로 C%p 변화했다.”

속도 개선 후에는:
“동일한 코퍼스와 모델에서 [개선 항목]을 적용해 검색·생성 구간의
응답시간 중앙값을 A초에서 B초로 줄였으며, 정답률은 X%에서 Y%로 변화했다.”

“정확도”가 무엇을 뜻하는지, 질문 수와 환경이 무엇인지 설명할 수 있어야 한다.
사전 실행 없이 예시 수치를 본인의 성과로 쓰지 않는다.
과거 GMS 서비스 성과로 소급하지 않고 후속 개선 경험으로 표현한다.

## 14. 진행 순서 체크리스트

1. Ollama 설치, 모델 2개 다운로드, GPU 적재 확인
2. Python 설치 복구, .venv-local 생성, 패키지 설치·버전 고정
3. dry-run으로 현재 정책의 501문서가 유지되는지 확인
4. 평가용 baseline DB 생성, smoke 3문항 실행
5. review.csv 채점, summarize 성공 확인
6. 에이전트가 준비한 개발셋·테스트셋과 정답 근거 검토
7. baseline 코퍼스에서 no_rag/rag 비교
8. 고정 데이터로 개발셋에서 검색·속도를 개선하고 테스트셋에서 최종 측정
9. 필요하면 실제 앱의 단계별 시간과 캐시 효과 측정
10. 원시 로그·결과표·실패 사례와 함께 경험 정리

실제 모델 호출이 성공하기 전까지는 GPU 추론 속도와 품질을 검증했다고 말하지 않는다.
이번 작업의 검증 범위는 도구 문법, CLI, 데이터 집계와 순수 집계 로직이다.



## 15. 개발셋 1차 검토 후 연결 주소 비교

2026-10-28: dev-001의 40개 응답을 에이전트가 채점·집계했다.
보고서는 eval/runs/dev-001/report.md에 있다.
초기 코드 보존본과 새 코드의 변경 이력은 eval/code_snapshots 및 eval/experiment-revisions.json에 기록했다.
초기 평가셋 manifest의 코드 해시는 변경 전 스냅샷을 가리키며, 질문·정답·기준은 그대로 유지한다.

모델·프롬프트·데이터·검색을 고정하고 연결 주소만 바꿔 다음 개발 실험을 실행한다.
새 --base-url 옵션은 선택 사항이며 기본값은 기존 localhost다.

```bash
python scripts/local_rag_eval.py run --dataset eval/dev.jsonl --corpus baseline --methods no_rag rag --repeats 1 --run-id dev-002 --base-url http://127.0.0.1:11434
```

이 실행은 40요청이다. 완료 후 에이전트에게 알리면 채점과 이전 실행 비교를 진행한다.
localhost/IPv4의 가벼운 연결 진단 차이를 모델 전체 응답시간 감소율로 그대로 사용하지 않는다.
최종 테스트는 개발셋에서 변경 사항을 확정한 뒤 진행한다.

## 16. dev-002 검토 후 응답 정책 비교

dev-002의 채점·주소 변경 비교 보고서는 eval/runs/dev-002/report.md에 있다.
개발셋에서 RAG 검색·생성 시간 중앙값은 4.617초에서 1.029초로 감소했다.
정답률은 15/20에서 14/20으로 달라져 모든 품질 지표 유지라고 결론내리지는 않는다.
검색 문서는 같았지만 생성 문장 일부가 달라졌다. 세부 판정은 보고서를 참고한다.

다음은 주소를 IPv4로 유지하고 응답 정책만 바꾸는 개발 실험이다.

```bash
python scripts/local_rag_eval.py run --dataset eval/dev.jsonl --corpus baseline --methods no_rag rag --repeats 1 --run-id dev-003 --base-url http://127.0.0.1:11434 --profile grounded
```

grounded는 원문 용어 보존, 질문한 사실만 최대 2문장 응답, 근거 없는 수치 억제를 적용한다.
baseline 프로필은 기존 정책이며 기본값으로 남아 있다.
원본 데이터·모델·검색·채점 기준은 변경하지 않았다.
새 프로필은 아직 모델로 실행·검증하지 않았고 결과를 보고 개선 여부를 판단한다.
사용자가 채점할 필요는 없다. 완료 후 에이전트에게 알려 결과 비교를 맡긴다.

## 17. 개발 단계 종료: 고정 테스트셋 최종 평가

dev-003 결과와 최종 명령은 eval/runs/dev-003/report.md에 있다.
개발셋에서 grounded RAG는 17/20 정답 기준 충족, 근거 미확인 추가 주장 응답은 2/20이었다.
잔여 오류를 기록한 상태에서 현재 정책을 고정해 최종 테스트로 넘어간다.
세 조건에서 각 360요청, 총 1,080요청이므로 이전 개발 실행보다 오래 걸린다.

Git Bash에서 순서대로 실행한다.

```bash
python scripts/local_rag_eval.py run --dataset eval/test.jsonl --corpus baseline --methods no_rag rag --repeats 3 --run-id final-001 --base-url http://localhost:11434 --profile baseline
python scripts/local_rag_eval.py run --dataset eval/test.jsonl --corpus baseline --methods no_rag rag --repeats 3 --run-id final-002 --base-url http://127.0.0.1:11434 --profile baseline
python scripts/local_rag_eval.py run --dataset eval/test.jsonl --corpus baseline --methods no_rag rag --repeats 3 --run-id final-003 --base-url http://127.0.0.1:11434 --profile grounded
```

A→B는 연결 주소, B→C는 응답 정책, A→C는 전체 변경의 효과를 비교한다.
각 조건 안에서 no_rag/rag도 비교한다.
질문 60개와 반복 3회를 구분하며, 반복을 독립 표본처럼 계산하지 않는다.
실행 도중 모델·코드·데이터·채점 기준을 변경하지 않는다.
결과 파일 경로를 알려주면 에이전트가 채점과 최종 결과 정리를 진행한다.

