import os
import shutil
import random
from tqdm import tqdm

def balance_dataset(input_folder, output_folder, balance_mode="equal", sample_size=None, train_ratio=0.8):
    """
    데이터셋 균형 조정 및 train/validation 분할
    
    - input_folder: 원본 데이터 폴더 (클래스별 폴더 구조)
    - output_folder: 데이터셋을 저장할 폴더
    - balance_mode: 
        - "equal": 모든 클래스를 동일한 개수로 맞춤 (기존 방식)
        - "original": 원본 데이터 비율 유지
    - sample_size: equal 모드에서 사용할 각 클래스당 샘플 개수
    - train_ratio: 학습/검증 데이터 분할 비율 (기본값: 0.8)
    """
    # ✅ 1. 클래스별 이미지 파일 목록 수집
    class_files = {}
    print("\n📊 원본 데이터 현황:")
    for class_name in os.listdir(input_folder):
        class_path = os.path.join(input_folder, class_name)
        if os.path.isdir(class_path):
            files = [os.path.join(class_path, f) for f in os.listdir(class_path) if f.endswith(('jpg', 'png', 'jpeg'))]
            class_files[class_name] = files
            print(f"- {class_name}: {len(files)}개")

    # train/validation 폴더 생성
    train_folder = os.path.join(output_folder, "train")
    val_folder = os.path.join(output_folder, "validation")
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)

    total_copied = 0
    for class_name, files in tqdm(class_files.items(), desc="클래스 처리 중"):
        if balance_mode == "equal":
            # 동일한 개수로 맞추기
            if sample_size is None:
                sample_size = min(len(files) for files in class_files.values())
            selected_files = random.sample(files, sample_size)
        else:  # "original" 모드
            selected_files = files

        # train/validation 분할
        random.shuffle(selected_files)
        split_idx = int(len(selected_files) * train_ratio)
        train_files = selected_files[:split_idx]
        val_files = selected_files[split_idx:]

        # train 폴더에 복사
        train_class_path = os.path.join(train_folder, class_name)
        os.makedirs(train_class_path, exist_ok=True)
        for file_path in train_files:
            shutil.copy(file_path, train_class_path)
            total_copied += 1

        # validation 폴더에 복사
        val_class_path = os.path.join(val_folder, class_name)
        os.makedirs(val_class_path, exist_ok=True)
        for file_path in val_files:
            shutil.copy(file_path, val_class_path)
            total_copied += 1

        print(f"\n📂 클래스: {class_name}")
        print(f"✅ Train 개수: {len(train_files)}장")
        print(f"✅ Validation 개수: {len(val_files)}장")
    
    print(f"\n✅ 데이터셋 생성 완료!")
    print(f"📁 저장 위치: {output_folder}")
    print(f"🔢 총 {total_copied}개 파일 복사됨")

# 사용 예제
input_folder = "../../data/good-data/cat/no-selected"
output_folder = "../../data/good-data/cat/selected"

# 방법 1: 모든 클래스를 동일한 개수로 맞추기
balance_dataset(input_folder, output_folder, balance_mode="equal", sample_size=264)

# 방법 2: 원본 데이터 비율 유지하기
# balance_dataset(input_folder, output_folder, balance_mode="original")