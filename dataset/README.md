# 운동 데이터 준비

`gym_exercises.csv`는 로컬 실행에 필요한 파일이며 이 저장소에는 포함하지 않습니다.
기존 로컬 CSV의 정확한 Kaggle 원본 URL, 작성자, 라이선스가 아직 기록되지 않았으므로, 원본 페이지와 이용·재배포 조건을 먼저 확인하세요.

확인한 CSV를 `dataset/gym_exercises.csv`에 배치하고 프로젝트 루트에서 `python 1_index_meta.py`를 실행합니다.
현재 코드는 다음 컬럼을 사용합니다.

- `Unnamed: 0`: 운동별 숫자 인덱스 (`source_id` 생성에 사용)
- `Title`, `Desc`, `Type`, `BodyPart`, `Equipment`, `Level`
- 기존 데이터에는 `Rating`, `RatingDesc`도 포함됩니다. 현재 전처리는 모든 컬럼을 대상으로 결측 행을 제외하므로 이 컬럼들도 처리 결과에 영향을 줍니다.

다른 데이터로 교체하면 기존 평가 질문의 운동 ID와 원문 근거가 맞지 않을 수 있습니다. 데이터 출처와 버전을 기록하고 평가 자료도 함께 점검하세요.
