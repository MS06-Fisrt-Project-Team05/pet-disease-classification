import os
import shutil
import random
from tqdm import tqdm

def balance_dataset(input_folder, output_folder, sample_size=None):
    """
    각 클래스 폴더에서 최소 개수만큼 랜덤 샘플링하여 새로운 폴더에 복사.
    
    - input_folder: 원본 데이터 폴더 (클래스별 폴더 구조)
    - output_folder: 균형 잡힌 데이터셋을 저장할 폴더
    - sample_size: 수동으로 지정한 샘플 개수 (기본값: 가장 작은 클래스 개수)
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

    # ✅ 2. 최소 개수 찾기 (샘플링할 개수 결정)
    if sample_size is None:
        sample_size = min(len(files) for files in class_files.values())
    
    print(f"\n⚡ 각 클래스별 {sample_size}개씩 샘플링하여 복사합니다.")

    # ✅ 3. 새로운 폴더에 균형 잡힌 데이터 저장
    os.makedirs(output_folder, exist_ok=True)
    
    total_copied = 0
    for class_name, files in tqdm(class_files.items(), desc="클래스 처리 중"):
        sampled_files = random.sample(files, sample_size)  # 랜덤 샘플링

        # 새로운 클래스 폴더 생성
        class_output_path = os.path.join(output_folder, class_name)
        os.makedirs(class_output_path, exist_ok=True)

        # 파일 복사
        for file_path in sampled_files:
            shutil.copy(file_path, class_output_path)
            total_copied += 1
    
    print(f"\n✅ 데이터 균형 조정 완료!")
    print(f"📁 저장 위치: {output_folder}")
    print(f"📈 총 {len(class_files)}개 클래스, 클래스당 {sample_size}개")
    print(f"🔢 총 {total_copied}개 파일 복사됨")

# 실행 예제
input_folder = "data/filtered_images/cat/validation"  # 원본 데이터 폴더 (클래스별 폴더 구조)
output_folder = "data/balanced_dataset/cat/validation"  # 새롭게 균형 맞춘 데이터 폴더
balance_dataset(input_folder, output_folder, sample_size=264)