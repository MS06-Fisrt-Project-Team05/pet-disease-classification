import os
import random
import shutil
from tqdm import tqdm

def balance_dataset(input_folder, output_folder, balance_mode="equal", sample_size=None):
    """
    train과 validation 폴더 내의 클래스별 데이터를 균형 조정
    
    - input_folder: 원본 데이터 폴더 (train과 validation 폴더, 그 아래 클래스별 폴더 구조)
    - output_folder: 데이터셋을 저장할 폴더
    - balance_mode: 
        - "equal": 모든 클래스를 동일한 개수로 맞춤
        - "original": 원본 데이터 비율 유지
    - sample_size: equal 모드에서 사용할 각 클래스당 전체 샘플 개수
    """
    # 입력 폴더 구조 확인
    train_folder = os.path.join(input_folder, "train")
    val_folder = os.path.join(input_folder, "validation")
    
    # 클래스 목록 가져오기 (train 폴더 기준)
    classes = [d for d in os.listdir(train_folder) if os.path.isdir(os.path.join(train_folder, d))]
    
    # 클래스별 파일 수집
    class_files = {}
    print("\n📊 원본 데이터 현황:")
    for class_name in classes:
        train_path = os.path.join(train_folder, class_name)
        val_path = os.path.join(val_folder, class_name)
        
        train_files = [os.path.join(train_path, f) for f in os.listdir(train_path) 
                      if f.endswith(('jpg', 'png', 'jpeg'))]
        val_files = [os.path.join(val_path, f) for f in os.listdir(val_path) 
                    if f.endswith(('jpg', 'png', 'jpeg'))]
        
        total_files = train_files + val_files
        class_files[class_name] = total_files
        print(f"- {class_name}: 총 {len(total_files)}개")

    # 출력 폴더 생성
    train_output = os.path.join(output_folder, "train")
    val_output = os.path.join(output_folder, "validation")
    os.makedirs(train_output, exist_ok=True)
    os.makedirs(val_output, exist_ok=True)

    total_copied = 0
    for class_name, files in tqdm(class_files.items(), desc="클래스 처리 중"):
        if balance_mode == "equal":
            if sample_size is None:
                # 각 클래스의 전체 데이터 중 최소 개수 찾기
                sample_size = min(len(files) for files in class_files.values())
            
            # 전체 파일에서 무작위로 sample_size만큼 선택
            selected_files = random.sample(files, sample_size)
        else:  # "original" 모드
            selected_files = files

        # 8:2 비율로 분할
        random.shuffle(selected_files)
        split_idx = int(len(selected_files) * 0.8)
        train_files = selected_files[:split_idx]
        val_files = selected_files[split_idx:]

        # train 폴더에 복사
        train_class_path = os.path.join(train_output, class_name)
        os.makedirs(train_class_path, exist_ok=True)
        for file_path in train_files:
            shutil.copy(file_path, train_class_path)
            total_copied += 1

        # validation 폴더에 복사
        val_class_path = os.path.join(val_output, class_name)
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
input_folder = "../../data/aihub/cat"  # train과 validation 폴더가 있는 경로
output_folder = "../../data/mixed-data/cat"

# 방법 1: 모든 클래스를 동일한 개수로 맞추기 (전체 데이터 기준 200개)
balance_dataset(input_folder, output_folder, balance_mode="equal", sample_size=200)

# 방법 2: 원본 데이터 비율 유지하기
# balance_dataset(input_folder, output_folder, balance_mode="original")