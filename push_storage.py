from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
import os
import glob  # 최신 JSON 파일 찾기 위한 라이브러리

# .env 파일에서 환경 변수 로드
load_dotenv()

# Azure Storage 설정
STORAGE_ACCOUNT_NAME = os.getenv("STORAGE_ACCOUNT_NAME")
STORAGE_ACCOUNT_KEY = os.getenv("STORAGE_ACCOUNT_KEY")
CONTAINER_NAME = "models"

# 최신 class_names.json 파일 찾기
class_files = sorted(glob.glob("class_names*.json"), reverse=True)
class_file = class_files[0] if class_files else "class_names.json"

# 업로드할 파일들
model_file = "resnet50_dog_disease.pth"

# Blob Storage 업로드 함수
def upload_to_blob(file_path, blob_name):
    try:
        blob_service_client = BlobServiceClient(
            account_url=f"https://{STORAGE_ACCOUNT_NAME}.blob.core.windows.net",
            credential=STORAGE_ACCOUNT_KEY
        )

        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        if not container_client.exists():
            container_client.create_container()

        blob_client = container_client.get_blob_client(blob=blob_name)

        with open(file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)

        print(f"✅ {blob_name} 업로드 완료!")
    except Exception as e:
        print(f"❌ {blob_name} 업로드 실패: {e}")

# 최신 모델과 클래스 정보 업로드
upload_to_blob(model_file, "resnet50_dog_disease.pth")
upload_to_blob(class_file, os.path.basename(class_file))  # 원본 파일명 유지

print("🎉 모든 파일 업로드 완료!")
