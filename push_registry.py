# 필요한 Azure ML SDK 및 기타 라이브러리 임포트
from azure.ai.ml import MLClient  # Azure ML 작업을 위한 주요 클라이언트
from azure.ai.ml.entities import Model  # 모델 엔티티 정의를 위한 클래스
from azure.identity import DefaultAzureCredential  # Azure 인증을 위한 클래스
from dotenv import load_dotenv  # 환경 변수 로드를 위한 라이브러리
import os  # 환경 변수 및 파일 경로 처리를 위한 라이브러리

# .env 파일에서 환경 변수 로드
load_dotenv()

def push_model_to_registry():
    # Azure ML 작업공간 접근에 필요한 환경 변수 가져오기
    subscription_id = os.getenv("SUBSCRIPTION_ID")  # Azure 구독 ID
    resource_group = os.getenv("RESOURCE_GROUP")    # 리소스 그룹 이름
    workspace_name = os.getenv("WORKSPACE_NAME")    # 작업공간 이름

    # DefaultAzureCredential을 사용하여 Azure 서비스 인증
    credential = DefaultAzureCredential()
    # Azure ML 클라이언트 객체 생성 및 작업공간 연결
    ml_client = MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name
    )

    # 등록할 모델 파일의 로컬 경로 지정
    # 모델 개발 후 이곳의 내용을 알맞게 수정합니다.
    model_path = "ResNet/model.pth"
    model_name = "resnet-model"
    model_description = "ResNet 모델"
    
    # 현재 등록된 모델의 최신 버전 확인
    try:
        models = ml_client.models.list(name=model_name)
        versions = [int(m.version) for m in models]
        next_version = str(max(versions) + 1) if versions else "1"
    except:
        next_version = "1"
    
    # 모델 엔티티 생성 및 메타데이터 설정
    model = Model(
        path=model_path,
        name=model_name,
        description=model_description,
        type="custom_model",
        version=next_version
    )

    # 모델을 Azure ML 모델 레지스트리에 등록
    registered_model = ml_client.models.create_or_update(model)
    # 등록 성공 메시지 출력
    print(f"모델이 성공적으로 등록되었습니다. 이름: {registered_model.name}, 버전: {registered_model.version}")

# 스크립트가 직접 실행될 때만 모델 등록 함수 실행
if __name__ == "__main__":
    push_model_to_registry()