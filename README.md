# 반려동물 피부질환 분류 AI 프로젝트

## 프로젝트 소개
이 프로젝트는 딥러닝을 활용하여 반려동물의 피부질환을 이미지로 분류하는 AI 모델을 개발하는 프로젝트입니다.

## 주요 파일 설명

### 데이터 전처리
- `filtering_data.py`: 이미지 품질 필터링
  - 블러 검사, 밝기 검사, 색상 다양성 검사 등을 통해 저품질 이미지 필터링
  - 중복 이미지 제거
  - 필터링된 이미지를 새로운 폴더에 저장

- `split_data.py`: 데이터셋 분할 및 균형 조정
  - 학습(train)/검증(validation) 데이터셋 분할
    - 기본 비율: 학습 80%, 검증 20%
    - train_ratio 매개변수로 비율 조정 가능
  
  - 클래스별 데이터 균형 조정 (2가지 모드 지원)
    1. equal 모드
       - 모든 클래스의 데이터 개수를 동일하게 맞춤
       - sample_size 매개변수로 각 클래스당 샘플 수 지정
       - 지정하지 않을 경우 가장 적은 클래스 기준으로 자동 설정
    
    2. original 모드
       - 원본 데이터의 클래스별 비율을 그대로 유지
       - 불균형 데이터셋 실험용
  
  - 데이터셋 처리 기능
    - 이미지 파일 자동 감지 (jpg, png, jpeg 지원)
    - 무작위 샘플링으로 편향 방지
    - 원본 파일 유지 (복사 방식)
  
  - 상세한 처리 현황 출력
    - 원본 데이터 클래스별 통계
    - 분할된 train/validation 데이터 개수
    - 전체 처리된 파일 수
    - 최종 저장 위치
  
  - 사용 예시:
    ```python
    # 방법 1: 클래스당 264개씩 균등 분할
    balance_dataset(
        input_folder="data/good-data/cat/no-selected",
        output_folder="data/good-data/cat/selected",
        balance_mode="equal",
        sample_size=264
    )

    # 방법 2: 원본 비율 유지
    balance_dataset(
        input_folder="data/good-data/cat/no-selected",
        output_folder="data/good-data/cat/selected",
        balance_mode="original"
    )
    ```

### 모델 학습
- `train_skin_disease.py`: 모델 학습 메인 스크립트
  - ResNet, EfficientNet 모델 지원
  - 개/고양이 데이터셋 선택 가능
  - 데이터 증강(augmentation) 적용
  - 학습 과정 모니터링 및 best 모델 저장
  - 데이터 경로 설정 필요:
    ```python
    # train_skin_disease.py 내부 경로 설정 예시
    # 방법 1: 기본 경로 사용
    base_path = f"../../data/good-data/{animal_type}/selected"
    
    # 방법 2: AIHub 원본 데이터 경로 사용
    base_path = f"../../data/aihub/{animal_type}"
    
    # 방법 3: 절대 경로 사용
    base_path = f"/home/user/projects/data/{animal_type}"
    ```
  - 실행 예시:
    ```bash
    python train_skin_disease.py --model resnet --animal dog
    or
    python train_skin_disease.py --model efficientnet --animal cat
    ```

### 모델 배포
- `push_registry.py`: Azure ML 모델 레지스트리 등록
  - 학습된 모델을 Azure ML 모델 레지스트리에 등록
  - 버전 관리 지원
  - 환경 변수를 통한 보안 설정

- `push_storage.py`: Azure Blob Storage 업로드
  - 모델 파일(.pth)과 클래스 정보(class_names.json) 업로드
  - Azure Blob Storage 컨테이너 자동 생성
  - 파일 중복 처리

### 기타 파일
- `requirements.txt`: 프로젝트 의존성 패키지 목록
- `class_names_*.json`: 클래스 레이블 정보
- `.gitignore`: Git 제외 파일 설정
- `.coderabbit.yaml`: CodeRabbit 설정

## 환경 설정
```bash
pip install -r requirements.txt
```

## 데이터 구조 (예시 실제 환경은 본인이 편한대로 구성)
```
data/
├── good-data/ <- 선별 데이터
│   ├── cat/
│   │   ├── selected/
│   │   └── no-selected/
│   └── dog/
│       ├── selected/
│       └── no-selected/
└── aihub/ <- 원본데이터
    ├── cat/
    └── dog/
```

## 주의사항
- 데이터 폴더(`data/`)는 용량 문제로 Git에서 제외됨
- Azure 관련 설정은 `.env` 파일에 저장 필요