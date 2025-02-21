CONTAINER_NAME="pet-skin-disease-data"  # 다운로드할 컨테이너 이름
LOCAL_DOWNLOAD_PATH="./data"  # 로컬 저장 경로
STORAGE_ACCOUNT="aihub0data"  # Storage 계정 이름

# 다운로드 폴더 생성
mkdir -p "$LOCAL_DOWNLOAD_PATH"

# 컨테이너 내 모든 Blob 다운로드
for blob in $(az storage blob list --account-name "$STORAGE_ACCOUNT" --container-name "$CONTAINER_NAME" --query "[].name" -o tsv); do
    local_path="$LOCAL_DOWNLOAD_PATH/$blob"  # 원래 폴더 구조 유지
    mkdir -p "$(dirname "$local_path")"  # 하위 디렉토리 생성
    az storage blob download --account-name "$STORAGE_ACCOUNT" --container-name "$CONTAINER_NAME" --name "$blob" --file "$local_path"
    echo "Downloaded: $local_path"
done
