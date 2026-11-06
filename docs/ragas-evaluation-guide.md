# 로컬 RAGAS 평가 실행 가이드

작성·공식 문서 확인: 2026-10-29 (Asia/Seoul)  
프로젝트: C:/tw/streamlit_lab  
대상: 기존 final-003 결과의 RAG 유무 비교 및 RAG 내부 품질 평가  
API 기준: ragas==0.4.3, ragas.metrics.collections

## 1. 이번에 확인할 것

다음 두 질문을 나눠 답한다.

1. RAG 효과: 동일 모델·질문·grounded 정책에서 자료를 제공하면 답변의 사실 일치도와 내용 충족도가 얼마나 달라지는가?
2. RAG 내부 품질: 필요한 내용을 검색했는가, 유용한 자료를 앞에 배치했는가, 답변의 주장이 실제 제공 자료로 뒷받침되는가?

기본 비교는 final-003/no_rag 대 final-003/rag다. 연결 주소 효과는 이번 평가의 주제가 아니다.
새로 챗봇 답변을 생성하지 않고 저장된 질문·답변·검색 문서를 평가용 모델에 넣는다.
기존 전처리·501문서·테스트·생성 프롬프트·원본 응답·기존 채점 점수는 유지한다.
이번 RAGAS 평가는 이미 결과를 확인한 뒤 추가하는 사후 보완 평가로 기록한다.

현재 상태: 전체 실행과 개발셋 평가자 검증을 수행했다. 판정 결함이 남아 확정 성과 수치는 보류한다. 제15절과 검증 보고서를 참고한다.

## 2. 지표와 적용 범위

| 목적 | 지표 | no_rag | rag | 주 평가 범위 |
|---|---|---|---|---|
| RAG 효과 | Factual Correctness: F1 | 적용 | 적용 | 답변 가능 45문항 |
| 정확성과 누락 구분 | Factual Precision / Recall | 적용 | 적용 | 답변 가능 45문항 |
| 질문 적합성 | AnswerRelevancy (문서상의 Response Relevancy) | 적용 | 적용 | 답변 가능 45문항 |
| 검색 순위 | ContextPrecisionWithReference | 해당 없음 | 적용 | 답변 가능 45문항 |
| 검색 정보 확보 | ContextRecall | 해당 없음 | 적용 | 답변 가능 45문항 |
| 생성 근거 충실성 | Faithfulness | 해당 없음 | 적용 | 답변 가능 45문항 |
| 올바른 보류 | 기존 답변 불가 판정·수치 조작·인용 검토 | 적용 | 적용 | 답변 불가 15문항 |

주 지표는 Factual F1로 정하고 Precision·Recall로 누락/추가 사실을 해석한다.
검색 지표와 Faithfulness는 RAG의 원인을 설명하는 별도 축이다.
일곱 지표를 평균 내어 하나의 “종합 RAGAS 정확도”로 만들지 않는다.

Factual Correctness는 답변과 reference의 사실을 비교한다. F1은 연속 점수이므로 필수 사실 하나가 빠졌다고 자동으로 문항 전체 0점이 되는 기존 방식과 다르다.
Precision은 reference가 지지하는 응답 사실의 비중, Recall은 응답에 포함된 reference 사실의 비중이다.
실제 사실 분해가 평가 모델에 의존하므로 세 개의 고정 사실로 항상 나뉜다고 보장하지 않는다.
출처: [Factual Correctness](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/factual_correctness/)

Faithfulness는 실제 retrieved_contexts에 대한 응답 주장 지지다. 정답에 필요한 내용을 누락해도 말한 내용만 모두 근거에 있으면 높을 수 있다.
기존 unsupported=0/1 응답 비율의 여집합과는 단위와 근거 범위가 다르다.
출처: [Faithfulness](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/)

Context Recall은 reference의 정보가 검색 자료에서 뒷받침되는지 보는 LLM 기반 지표다.
기존 relevant_groups의 문서 ID 기반 Recall@5=94.4%와 정의가 다르다.
Context Precision은 유용하다고 판정된 문서의 순서를 반영한 평균 정밀도 방식이다.
“관련 문서 수 / 5”인 단순 Precision@5와 혼동하지 않는다.
출처: [Context Recall](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_recall/),
[Context Precision](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/)

AnswerRelevancy는 응답으로부터 질문을 생성하고 원 질문과 임베딩 유사도를 비교한다.
사실 정확도 지표는 아니며, 평가용 LLM 외에 임베딩 모델도 필요하다.
출처: [Response Relevancy](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/answer_relevance/)

## 3. 답변 가능 45문항과 답변 불가 15문항을 분리하는 이유

final-003의 60문항 × 두 방법 × 세 반복은 총 360개 응답이다.

- RAGAS 주 분석: 답변 가능 45문항 × 두 방법 × 세 반복 = 270개 응답.
- 방법별 비교: 135개 응답, 동일 question_id/repeat의 135쌍.
- 답변 불가 분석: 15문항 × 두 방법 × 세 반복 = 90개 응답을 기존 기준으로 별도 보고한다.

AnswerRelevancy는 보류성 답변에 낮은 점수를 줄 수 있고, Faithfulness는 주장 없는 응답에서 해석이 어렵거나 비정상 점수가 나올 수 있다.
따라서 “개인 kg를 알 수 없다”는 올바른 답변을 일반 지표의 낮은 점수만으로 오답으로 바꾸지 않는다.
이 동작은 [0.4.3 AnswerRelevancy 구현](https://github.com/vibrantlabsai/ragas/blob/v0.4.3/src/ragas/metrics/collections/answer_relevancy/metric.py)에서도 확인할 수 있다.

기존 39.4%→72.2%는 답변 불가를 포함한 60문항의 엄격 완전정답률이다.
새 주 분석의 F1은 45문항의 연속 점수다. 두 숫자를 같은 정확도처럼 직접 비교하지 않는다.
기존 결과는 eval/final-report.md에 보존돼 있다.

## 4. 사용하는 파일과 필드

| 기존 기록 | RAGAS 입력 |
|---|---|
| eval/test.jsonl의 question | user_input |
| 각 run answers.jsonl의 answer | response |
| test.jsonl의 reference_answer | reference |
| 해당 응답의 retrieved_documents[].text | retrieved_contexts |
| question_id, method, repeat | 조건 간 대응·집계 키 |
| reference_evidence, required_facts | 판정 불일치 검토용 보조 정보 |

retrieved_contexts에는 당시 모델에 실제 제공된 문서와 순서를 넣는다.
새로 검색하거나 reference_evidence를 검색 결과처럼 넣지 않는다.
문서별 [근거 N] 표시는 실제 프롬프트와 같은 순서로 유지한다.
no_rag의 retrieved_contexts는 빈 목록으로 남기며 검색·Faithfulness를 0점 처리하지 않는다.

보조 스크립트: scripts/local_ragas_eval.py

- prepare --check: 해시·행 수·문항·실제 본문 일치를 읽기 전용으로 검증.
- prepare: RAGAS 입력 JSONL과 원본 해시 manifest 생성.
- score: 저장 답변을 로컬 평가 모델로 채점. 챗봇 답변 생성 명령과 별개.
- summarize: 지표별 유효 점수·실패·짝 비교 결과 저장.

## 5. 평가 모델과 환경

| 역할 | 현재/예시 설정 |
|---|---|
| 평가 대상 생성 모델 | qwen3:8b-q4_K_M, 기존 생성 결과 고정 |
| 평가 대상 검색 임베딩 | bge-m3, 기존 검색 결과 고정 |
| RAGAS 평가용 LLM | 별도 설정. 우선 설치돼 있는 Qwen으로 연결·탐색 점검 가능 |
| AnswerRelevancy 평가용 임베딩 | bge-m3, 모든 비교 조건에서 동일하게 사용 |

같은 Qwen을 평가자로 쓰면 생성 모델과 평가 모델이 같다는 한계를 기록한다.
한국어·영어 의미 판정, 근육명 구분, 누락 탐지, JSON 출력을 개발셋에서 점검한다.
같은 모델의 점수만으로 독립 검증이나 사용자 만족도를 주장하지 않는다.
별도 평가 모델을 사용한다면 설치된 로컬 모델을 선택하고 이름·digest·옵션을 고정한다.
이 가이드는 추가 모델 다운로드나 유료 API 사용을 전제로 하지 않는다.

개발셋의 기존 review.csv·agent-review.json과 평가 결과를 Codex가 대조한다.
상복부/복사근, 이두근/삼두근, 전완근/종아리, 반복수/중량 같은 혼동을 확인한다.
완전한 답변·일부 누락·명백한 오답을 구분하는지, Faithfulness가 부분답변에도 높을 수 있는지를 점검한다.
사용자에게 전체 수동 채점을 요구하는 단계는 아니다.

## 6. Git Bash에서 설치 준비

기존 (.venv)을 사용한다. 새 터미널이라면 활성화한다.

```bash
cd /c/tw/streamlit_lab
source .venv/Scripts/activate
python --version
ollama list
mkdir -p eval/ragas
export RAGAS_DO_NOT_TRACK=true
```

### 현재 프로젝트에서 확인한 의존성 충돌과 해결

원본 환경 목록은 eval/ragas/environment-before-ragas.txt에 보존한다.
원본 snapshot 전체를 그대로 -c로 지정하면 다음 네 고정값을 조정해야 한다. 앞의 세 개는 패키지 요구 범위 충돌이고, 마지막은 실제 import 경로 호환 문제다.

| 기존 패키지 | 기존 버전 | 검증한 호환 버전 | 제약을 요구하는 패키지 |
|---|---|---|---|
| fsspec | 2026.9.0 | 2026.7.0 | datasets 5.1.0 |
| jiter | 0.17.0 | 0.14.0 | instructor 1.17.0 |
| rich | 15.0.0 | 14.3.4 | instructor 1.17.0 |
| langchain-community | 0.4.2 | 0.4.1 | ragas 0.4.3의 VertexAI import |

2026-10-29 현재 .venv에서 수정 제약 파일을 사용한 pip dry-run이 성공했다.
설치 전 pip check도 통과했고, 설치 예정 조합을 전체 설치 패키지의 요구 범위와 대조해 충돌이 없음을 확인했다.
사용자가 설치를 진행한 뒤 langchain-community 0.4.2의 VertexAI 모듈 삭제로 import 오류가 확인됐다.
0.4.1 공식 wheel을 임시 경로에 올려 실제 RAGAS import, llm_factory/embedding_factory 생성, 7개 지표 객체 생성까지 성공했다.
이 검증은 실제 설치를 바꾸거나 모델을 호출하지 않았다. 현재 설치에 적용하려면 아래 호환성 수정 명령을 실행한다.
기록: eval/ragas/import-compatibility-check.json.

사용할 파일은 eval/ragas/constraints-ragas.txt다.
이 파일은 기존 snapshot에서 위 네 패키지의 고정값을 조정하고,
새로 필요한 패키지를 성공한 설치 계획의 버전으로 고정한 현재 프로젝트용 파일이다.
원본 environment-before-ragas.txt를 수정하거나 다시 생성하지 않는다.
모델·임베딩·원본 CSV·501문서·기존 평가 결과는 이 설치 작업과 별개로 유지한다.

처음 준비할 때 원본 snapshot이 없다면 다음처럼 한 번만 남긴다.

```bash
if [ ! -f eval/ragas/environment-before-ragas.txt ]; then
  python -m pip freeze > eval/ragas/environment-before-ragas.txt
fi
```

현재 프로젝트에는 snapshot과 검증된 constraints-ragas.txt가 모두 준비돼 있다.
다른 PC나 다른 환경에서는 기존 파일을 무조건 재사용하지 않고 설치 계획을 다시 검증한다.

의존성 해결을 다시 확인하려면 다음을 실행한다. --dry-run은 설치를 변경하지 않는다.

```bash
python -m pip install --dry-run -c eval/ragas/constraints-ragas.txt "ragas==0.4.3"
```

실제 설치와 확인:

```bash
python -m pip install -c eval/ragas/constraints-ragas.txt "ragas==0.4.3"
python -m pip check
python -m pip freeze > eval/ragas/environment-after-ragas.txt
python -c "from importlib.metadata import version; print(version('ragas')); from ragas.metrics.collections import FactualCorrectness, AnswerRelevancy, ContextPrecisionWithReference, ContextRecall, Faithfulness; print('imports OK')"
```

성공 시 ragas 버전 0.4.3, No broken requirements found., imports OK를 확인한다.
검증 기록은 eval/ragas/install-plan-ragas.json과 install-plan-ragas-summary.json에 있다.
최초 환경 기준 변경은 기존 네 패키지 조정과 새 패키지 16개 추가다.
RAGAS를 이미 설치한 현재 단계의 추가 수정은 langchain-community 0.4.2→0.4.1 한 패키지다.
기존 Python/가상환경 삭제·재생성 없이 진행하는 방법이다.
추가 충돌이나 import 오류가 나오면 출력을 바탕으로 Codex가 이어서 확인한다.
루트 requirements.txt는 이번 가이드 작성에서 변경하지 않았다.

constraints는 요구사항을 무시하는 옵션이 아니라 허용 버전을 제한한다.
상충하는 기존 고정값을 그대로 둔 채 새 범위를 추가하면 해결되지 않는다.
[pip 의존성 해결 공식 설명](https://pip.pypa.io/en/stable/topics/dependency-resolution/)

확인한 배포 버전과 API: [PyPI ragas 0.4.3](https://pypi.org/project/ragas/0.4.3/).
collections API에 기존 LangchainLLMWrapper/LangchainEmbeddingsWrapper를 그대로 혼합하지 않는다.
[0.4.3 컴포넌트 검증 코드](https://github.com/vibrantlabsai/ragas/blob/v0.4.3/src/ragas/metrics/collections/base.py)

### RAGAS 설치 후 VertexAI import 오류가 난 경우

다음 오류는 RAGAS 0.4.3이 삭제된 langchain-community import 경로를 사용하는 호환 문제다.

```text
ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'
```

공식 저장소에도 같은 문제가 보고돼 있다:
[RAGAS issue #2745](https://github.com/vibrantlabsai/ragas/issues/2745).
현재 프로젝트에서는 공식 배포 파일과 임시 import 검증으로 langchain-community==0.4.1 조합을 확인했다.
이 버전은 현재 langchain-core 1.6.12와 langchain-classic 1.0.8의 요구 범위에도 맞는다.

이미 RAGAS 설치를 마쳤다면 다음 명령으로 호환성을 수정한다.

```bash
python -m pip install -c eval/ragas/constraints-ragas.txt "langchain-community==0.4.1"
python -m pip check
python -c "from ragas.metrics.collections import FactualCorrectness, AnswerRelevancy, ContextPrecisionWithReference, ContextRecall, Faithfulness; print('imports OK')"
python -m pip freeze > eval/ragas/environment-after-ragas.txt
```

최종 dry-run에서 기존 패키지 변경은 langchain-community 하나였다.
의존성 dataclasses-json, marshmallow, typing-inspect, mypy_extensions 네 개가 새로 설치된다.
여기까지 통과하면 7번 개발셋 점검으로 이어간다.

## 7. 개발셋으로 연결과 판정 점검

먼저 입력 검증을 수행하고 개발셋용 파일을 준비한다.
prepare는 이미 존재하는 입력 파일을 덮어쓰지 않는다.

```bash
python scripts/local_ragas_eval.py prepare --source-run dev-003 --dataset eval/dev.jsonl --check

python scripts/local_ragas_eval.py prepare --source-run dev-003 --dataset eval/dev.jsonl --out eval/ragas/dev-003.samples.jsonl
```

설치돼 있는 모델로 우선 소량 실행한다. 아래 Qwen 설정은 탐색용 예시다.
이 명령부터 평가용 LLM 호출이 발생한다.

```bash
export RAGAS_JUDGE_MODEL="qwen3:8b-q4_K_M"

python scripts/local_ragas_eval.py score \
  --input eval/ragas/dev-003.samples.jsonl \
  --out eval/ragas/pilot-dev-001 \
  --judge-model "$RAGAS_JUDGE_MODEL" \
  --questions d001 d002 d016 \
  --metrics factual_f1 faithfulness

python scripts/local_ragas_eval.py summarize --out eval/ragas/pilot-dev-001
```

그다음 임베딩을 쓰는 지표와 검색 지표까지 점검한다.

```bash
python scripts/local_ragas_eval.py score \
  --input eval/ragas/dev-003.samples.jsonl \
  --out eval/ragas/pilot-dev-002 \
  --judge-model "$RAGAS_JUDGE_MODEL" \
  --questions d001 d002 d016

python scripts/local_ragas_eval.py summarize --out eval/ragas/pilot-dev-002
```

기본 평가 주소는 http://127.0.0.1:11434/v1 이다. 기존 생성 API 주소와 달리 /v1을 포함한다.
OpenAI SDK를 로컬 Ollama에 연결하며 api_key의 ollama는 로컬 연결용 더미 값이다.
공식 연결 예시: [Ragas Quick Start](https://docs.ragas.io/en/stable/getstarted/quickstart/),
[Ollama OpenAI compatibility](https://docs.ollama.com/api/openai-compatibility).

평가자는 temperature=0, top_p=1, seed=42, max_tokens=4096, reasoning_effort=none을 기본값으로 사용한다.
RAGAS 지표의 JSON 출력에 맞춰 설정한 평가자 옵션이며 기존 챗봇 생성 옵션을 바꾸는 것이 아니다.
한국어·영어 평가 지시는 스크립트에 고정돼 있고 config.json에 저장된다.

JSON 파싱·컨텍스트 길이·thinking 호환성 문제가 있으면 개발 단계에서 해결하고 평가자 설정을 고정한다.
필요하면 --reasoning-effort unset 등 호환 설정을 새 파일에 시험한다. 설정을 바꾼 결과는 새 출력 폴더로 분리한다.
평가 모델의 컨텍스트 크기도 제공 자료와 스키마를 수용하는지 확인한다.
출력 제한을 늘리는 것과 입력 컨텍스트 크기를 늘리는 것은 별개다.

## 8. 최종 결과 준비와 두 축 평가

개발 단계에서 평가자의 작동과 판정이 적절한지 확인한 후 실행한다.
모델 이름만으로 평가자 타당성이 확인되는 것은 아니다.

```bash
python scripts/local_ragas_eval.py prepare --check
python scripts/local_ragas_eval.py prepare
```

정상 준비 시 60문항·360행, 답변 가능 270행·답변 불가 90행이 표시된다.
파일은 eval/ragas/final-003.samples.jsonl과 같은 위치의 manifest로 생성된다.

아래 기본 실행은 두 방법의 공통 지표와 RAG 전용 지표를 모두 계산한다.
답변 불가 문항은 기본적으로 제외한다.

```bash
python scripts/local_ragas_eval.py score \
  --input eval/ragas/final-003.samples.jsonl \
  --out eval/ragas/final-003-judge-001 \
  --judge-model "$RAGAS_JUDGE_MODEL"

python scripts/local_ragas_eval.py summarize --out eval/ragas/final-003-judge-001
```

기본 지표 7개에서 예상되는 논리적 결과 수는 다음과 같다.

- no_rag: 공통 지표 4개 × 135응답 = 540개 유효 점수.
- rag: 지표 7개 × 135응답 = 945개 유효 점수.
- no_rag의 RAG 전용 지표: 3개 × 135응답 = 405개 해당 없음.
- 합계: 유효 1,485개 + 해당 없음 405개 = 결과 키 1,890개.

이는 LLM API 호출 횟수가 아니다. 한 지표가 사실 분해·검증·역질문 생성 등 여러 호출을 할 수 있다.
동일 질문·reference·답변 또는 동일 검색 입력이 반복되면 정확히 같은 지표 입력의 결과를 재사용한다.
따라서 실제 평가 호출은 캐시와 중복 정도에 따라 달라진다.
평가 호출의 소요 시간은 기존 챗봇 응답 지연과 구분한다.

## 9. 오류·중단·재개

출력 폴더가 이미 있으면 새 실행은 거부한다.
score는 지표별 결과를 scores.jsonl에 즉시 기록하고, 기본적으로 첫 오류에서 멈춘다.
평가 오류를 0점이나 성공 응답으로 바꾸지 않는다.

기술적 오류를 해결한 뒤 같은 설정으로 재개할 때는 원래 명령에 --resume을 추가한다.

```bash
python scripts/local_ragas_eval.py score \
  --input eval/ragas/final-003.samples.jsonl \
  --out eval/ragas/final-003-judge-001 \
  --judge-model "$RAGAS_JUDGE_MODEL" \
  --resume
```

이미 성공한 결과를 보존하고 미완료/실패 항목만 이어간다.
실패 기록도 JSONL에 남아 technical_failure_records로 집계된다.
설정·입력 해시·패키지 버전·스크립트·모델 digest가 달라지면 같은 폴더의 재개를 거부한다.
잘 나온 점수만 고르기 위한 재시도는 하지 않는다.
기술적 실패 사유와 수정 사항을 기록한다.

파일의 마지막 JSONL 행이 손상된 경우에는 원본을 보존하고 Codex가 먼저 확인한다.
평가 실패가 남으면 유효 수/예정 수와 대응 비교에 포함된 수를 공개하고 최종 성과로 확정하지 않는다.
--continue-on-error는 오류를 모으는 진단용 선택지이며 오류가 사라지는 옵션이 아니다.

### token repeat limit reached로 중단된 경우

이 오류는 평가 모델의 출력 반복 때문에 Ollama가 요청을 중단한 경우다.
저장 파일이 유효하면 기존 성공 점수를 유지하고 같은 옵션으로 --resume을 한 번 시도할 수 있다.
실패 기록은 남고 그 항목이 먼저 재평가된다. 실행이 끝나기 전 scoring_complete=false는 미완료 표시다.
같은 항목에서 반복 실패하면 로그를 보존하고 Codex가 추가 진단한다.
재개 전 평가 스크립트나 평가자 옵션을 바꾸면 원래 config의 해시 검증이 거부할 수 있다.
평가자 옵션을 바꿔야 한다면 기존 결과와 구분된 계획으로 처리한다.

### 같은 항목에서 반복 실패할 때 나머지 진행

모델 메모리 초기화 후에도 같은 요청이 실패하면, 같은 원래 옵션에서 오류만 기록하고 다음 항목으로 진행할 수 있다.
아래 옵션은 기존 코드가 지원하며 평가 모델·프롬프트·점수 정책을 바꾸지 않는다.

```bash
python scripts/local_ragas_eval.py score \
  --input eval/ragas/final-003.samples.jsonl \
  --out eval/ragas/final-003-judge-001 \
  --judge-model "qwen3:8b-q4_K_M" \
  --resume --continue-on-error &&
python scripts/local_ragas_eval.py summarize --out eval/ragas/final-003-judge-001
```

이 명령은 오류를 해결하거나 0점으로 처리하는 명령이 아니다.
완료 항목을 유지하고 미완료/오류 항목을 재시도하되, 실패해도 나머지를 진행한다.
오류가 남으면 scoring_complete=false가 유지된다. 최종 분석에는 유효 수·실패 수·대응 가능한 쌍을 표시한다.
완전한 최종 점수는 남은 실패의 해결과 평가자 판정 검토 후 확정한다.
평가 설정을 바꿀 경우 새 run을 사용해 서로 다른 평가자의 점수를 임의로 섞지 않는다.

## 10. 산출물과 집계 해석

각 평가 출력 폴더에는 다음 파일이 생긴다.

- config.json: 입력·스크립트 해시, 라이브러리 버전, 평가 모델/digest, 지표 옵션.
- source-manifest.json: 기존 생성 run의 설정·답변·데이터 해시.
- scores.jsonl: sample_id, question_id, method, repeat, 지표, 점수, 상태, 오류, 재사용 출처.
- summary.json: 지표별 평균·유효 수·미완료 수와 RAG/no_rag의 동일 문항·반복 대응 차이.

RAGAS가 reason/traces를 반환하면 저장한다. 기본 지표가 중간 사실 판정이나 이유를 반환하지 않으면 null일 수 있다.
숫자만으로 해석하기 어려운 경우 원래 질문·답변·원문 및 기존 판정 이유와 대조한다.
이 스크립트가 모든 내부 LLM 호출의 완전한 원시 추적을 제공하는 것은 아니다.

scoring_complete=true와 각 지표의 expected/valid가 일치하는지 확인한다.
RAG 전용 지표의 no_rag expected=0/not_applicable=135는 정상이다.
실패 항목을 제외한 평균은 mean_valid_attempts라는 이름으로 보이며 그 수를 함께 공개한다.
주 보고는 문항별 세 반복 평균을 먼저 구한 mean_question_means와 대응 비교를 사용한다.
같은 질문 세 반복을 독립 질문 세 개로 취급하지 않는다.
135쌍은 45개 질문의 세 반복이다. 신뢰구간을 계산하면 기존 출처 연결 군집과 조건 대응을 유지한다.

Factual Precision/Recall/F1은 각각 별도 RAGAS 호출이므로 평가자의 사실 분해 변동으로
개별 실행의 F1이 별도로 계산된 Precision·Recall의 수식값과 정확히 같지 않을 수 있다.
평균 Precision·평균 Recall로 F1을 다시 만들어 원래 지표를 대체하지 않는다.
F1을 주 지표로 고정하고 큰 불일치는 개발 점검과 사례 분석에서 확인한다.

Factual Precision은 좁은 reference에 없는 참인 배경 설명도 낮게 평가할 수 있다.
그런 경우 실제 검색 자료가 지지하는지 Faithfulness와 원문 대조로 구분한다.
reference 자체의 누락·질문 범위 불일치도 자동으로 해결되지 않는다.
기존 정답을 바꿔야 한다면 별도의 v2 기준으로 기록하고 양쪽 방법을 같은 기준으로 다시 평가한다.

## 11. 답변 불가와 기존 검토를 함께 보고하기

15문항의 개인 중량·회복 일수·효과 보장·개인 이력 질문은 기존 review.csv의 판정을 별도 표로 보고한다.
최종 grounded 조건의 기존 결과는 다음과 같다.

| 기존 엄격 평가 항목 | no_rag | rag |
|---|---:|---:|
| 전체 60문항×3 완전정답 | 71/180 (39.4%) | 130/180 (72.2%) |
| 답변 가능 45문항×3 완전정답 | 26/135 (19.3%) | 88/135 (65.2%) |
| 답변 불가 15문항×3 완전정답 | 45/45 (100%) | 42/45 (93.3%) |

인용 번호 조작, 시스템 지시문 재출력, 근거 없이 만든 개인 수치도 기존 검토에서 계속 확인한다.
높은 RAGAS 점수가 이런 제품 수준 결함을 모두 검증해 주는 것은 아니다.

## 12. 결과 보고와 자소서에 쓰는 방법

RAGAS 결과가 나오면 아래를 실제 수치로 채운다. 현재 빈칸은 측정값이 아니다.

| 비교 항목 | no_rag | rag | 차이 |
|---|---|---|---|
| Factual F1: 45문항 | 측정 후 기입 | 측정 후 기입 | 대응 평균 차이 |
| Factual Precision | 측정 후 기입 | 측정 후 기입 | 대응 평균 차이 |
| Factual Recall | 측정 후 기입 | 측정 후 기입 | 대응 평균 차이 |
| AnswerRelevancy | 측정 후 기입 | 측정 후 기입 | 대응 평균 차이 |
| Context Precision | 해당 없음 | 측정 후 기입 | 내부 진단 |
| Context Recall | 해당 없음 | 측정 후 기입 | 내부 진단 |
| Faithfulness | 해당 없음 | 측정 후 기입 | 내부 진단 |

연속 점수 0.9를 “사용자 정답률 90%”라고 쓰지 않는다.
F1의 개선은 0~1 점수 차이로 쓰고, 100점 환산 시 몇 점 차이인지 명시한다.
기존 완전정답률의 +32.8%p와는 다른 지표다.

보고 문장 예시:
“로컬 501문서 RAG의 효과를 동일 모델·질문·응답 정책에서 비교하고,
RAGAS의 사실 일치도와 검색·근거 충실성 지표를 분리 평가했다.
답변 가능 45문항의 Factual F1은 [측정값]→[측정값]이었으며,
누락·번역·검색 실패와 답변 불가 문항의 보류 성능을 별도로 분석했다.”

같은 Qwen 평가자를 썼다면 모델 이름과 자기 평가 한계를 함께 기록한다.
이 점수는 로컬 CSV-RAG 후속 평가이며 실제 사용자 만족도·전체 앱·과거 GMS 서비스의 측정값은 아니다.
평가 모델의 지표를 사용했다는 사실과 독립 전문가 검증을 구분한다.
사용자가 실행을 완료하면 Codex가 입력·설정·실패·사례를 확인하고 집계와 문장 작성을 이어간다.

## 13. 완료 순서

1. 환경 snapshot 보존, ragas==0.4.3 설치와 import 확인.
2. dev-003 소량 평가 후 Codex가 기존 원문 판정과 비교.
3. 평가자·임베딩·옵션을 고정.
4. final-003 저장 결과로 두 축 평가.
5. summarize의 완결성과 지표별 분모 확인.
6. 15개 답변 불가 문항·인용·지시문 재출력 결과를 별도 결합.
7. RAG 효과, 내부 실패 원인, 판정 한계를 함께 보고.

이 단계에서 기존 local_rag_eval.py run 명령을 다시 실행할 필요는 없다.

## 14. 첫 전체 실행 후 검증 결과 (2026-10-29)

final-003-judge-001은 1,890개 키를 모두 처리했지만 t020의 Recall/F1 각 세 반복, 총 6개가 실패로 남았다.
또한 CSV 값과 명백히 일치하는 답변이 F1=0으로 판정되는 사례가 확인됐다.
t003의 사실 Precision은 동일 설정 단건 진단에서 원래 0과 달리 1이 나왔다.
원래 점수는 그대로 보존했다. 현재 결과는 탐색용이며 평가자 안정성 검증 전 성과 수치를 확정하지 않는다.
자세한 수치·유효 분모·사례·다음 단계는 eval/ragas/final-003-judge-001/report.md에 있다.
기존 답변을 처음부터 다시 생성하거나 같은 재개 명령을 계속 반복할 단계는 아니다.


## 15. 개발셋 평가자 검증 완료 (2026-10-29)

[평가자 검증 보고서](../eval/ragas/judge-validation-001/report.md)를 확인한다.
개발 8문항 본 검증 156개와 기준 답변 보완 진단 40개를 모두 실행했다. 현재 구성의 사실 판정·대상 문서 귀속·반복 안정성에 결함이 재현됐다.
기술적 오류는 0개였지만 의미 판정 검증을 통과하지 못했으므로 기존 전체 실행의 점수를 확정 성과로 사용하지 않는다.
기준 답변에 운동명을 명시하면 일부 정답 감점이 줄었지만 무관 문서의 같은 난이도를 다른 운동의 근거로 인정하는 문제와 근육명 판정이 남았다.
모호한 정답 기준은 review-findings.json에 이유를 기록했다. 원래 점수와 고정 테스트셋은 유지했다.
수정은 개발셋에서 검증하고, 미사용 개발 문항으로 확인한 후 새 평가 버전으로 전체 비교한다. 기존 원본에 수동 점수를 끼워 넣거나 높은 재시도 값으로 덮어쓰지 않는다.
