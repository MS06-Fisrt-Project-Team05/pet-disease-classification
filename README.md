# 반려동물 피부질환 분류 AI 프로젝트

## 프로젝트 소개

이 프로젝트는 딥러닝을 활용하여 반려동물의 피부질환을 이미지로 분류하는 AI 모델을 개발하는 프로젝트입니다. 다양한 이미지 분류 모델을 실험하고 최적의 성능을 도출하는 것을 목표로 합니다.

## 현재 지원하는 피부질환 분류

- 구진/플라크
- 비듬/각질/상피성잔고리
- 태선화/과다색소침착
- 농포/여드름
- 미란/궤양
- 결절/종괴

## 프로젝트 구조

```
├── ResNet/ # ResNet 모델 개발 폴더
│ ├── ClassifyPetDisease.ipynb -> 이름이 중복되면 수정
│ └── model.pth
├── VGG/ # VGG 모델 개발 폴더(다른 이미지 분류 모델 예시)
| ├── ClassifyPetDisease.ipynb -> 이름이 중복되면 수정
│ └── model.pth
|
├── push_registry.py # Azure Model Registry 업로드 코드
└── ...
```

## 협업 가이드

### 0. 환경 설정

```
pip install azure-ai-ml azure-identity
```

### 1. 모델 개발

1. data 는 최상위 폴더에 data 폴더로 추가해서 사용해주세요. 용량이 커서 github에는 올리지 않습니다.
2. 본인이 개발할 모델의 이름으로 새로운 폴더를 생성합니다 (예: `ResNet`, `VGG` 등)
3. 해당 폴더 내에서 모델 개발 및 학습을 진행합니다
4. 학습이 완료된 모델은 `.pth` 형식으로 저장합니다

### 2. Azure Model Registry 등록

1. 모델 개발이 완료되면 `push_registry.py` 파일을 수정합니다
2. 다음 정보들을 업데이트합니다:
   - 모델 경로
   - 모델 이름
   - 설명
3. `push_registry.py` 실행하여 Azure Model Registry에 모델을 등록합니다

```
python push_registry.py
```

### 3. 문서화

각 모델 폴더에는 다음 내용을 포함하는 문서화가 필요합니다:

- 모델 설명
- 학습 데이터셋 정보
- 하이퍼파라미터 설정
- 성능 메트릭스 (정확도, 손실 등)
- 학습 결과 그래프
