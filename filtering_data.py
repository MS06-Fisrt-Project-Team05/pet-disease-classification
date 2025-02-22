import cv2
import numpy as np
import os
from tqdm import tqdm
import hashlib

# ✅ 블러 검사 (Laplacian)
def is_blurry(image, threshold=100):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold  # 임계값보다 작으면 블러

# ✅ 밝기 검사 (평균 밝기)
def is_too_dark(image, threshold=50):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    return mean_brightness < threshold  # 임계값보다 작으면 너무 어두움

# ✅ 색상 다양성 검사 (히스토그램)
def is_low_color_variance(image, threshold=0.02):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0], None, [256], [0, 256])
    hist /= hist.sum()
    entropy = -np.sum(hist * np.log2(hist + 1e-8))
    return entropy < threshold  # 엔트로피가 낮으면 색상 다양성이 낮음


def md5_hash(image):
    return hashlib.md5(image.tobytes()).hexdigest()

def is_exact_duplicate(image, seen_md5_hashes):
    """
    픽셀 단위로 동일한 이미지를 필터링.
    이미 저장된 MD5 해시 목록(seen_md5_hashes)과 비교해
    동일하면 True, 아니면 False
    """
    h = md5_hash(image)
    if h in seen_md5_hashes:
        return True
    seen_md5_hashes.add(h)
    return False

# ✅ 데이터 필터링 실행
def filter_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print(f"\n📁 입력 폴더: {input_folder}")
    print(f"📁 출력 폴더: {output_folder}")

    total_images = len(os.listdir(input_folder))
    filtered_out = 0
    saved_images = 0

    # 중복 체크를 위한 MD5 해시 저장소
    seen_md5_hashes = set()

    for img_name in tqdm(os.listdir(input_folder)):
        img_path = os.path.join(input_folder, img_name)
        image = cv2.imread(img_path)
        if image is None:
            print(f"⚠️ 이미지 로드 실패: {img_name}")
            filtered_out += 1
            continue

        # 1) **완전히 동일한 이미지인지 검사**
        if is_exact_duplicate(image, seen_md5_hashes):
            print(f"❌ 중복(동일) 이미지 제외: {img_name}")
            filtered_out += 1
            continue

        # 4) **블러 검사**
        if is_blurry(image):
            print(f"❌ 블러 이미지 제외: {img_name}")
            filtered_out += 1
            continue

        # 5) **너무 어두운지 검사**
        if is_too_dark(image):
            print(f"❌ 너무 어두운 이미지 제외: {img_name}")
            filtered_out += 1
            continue

        # 6) **색상 다양성 검사**
        if is_low_color_variance(image):
            print(f"❌ 낮은 색상 다양성 이미지 제외: {img_name}")
            filtered_out += 1
            continue

        # ✅ 모든 필터 통과 시 저장
        output_path = os.path.join(output_folder, img_name)
        cv2.imwrite(output_path, image)
        saved_images += 1

    # ✅ 최종 통계 출력
    print(f"\n📊 처리 완료:")
    print(f"- 전체 이미지: {total_images}개")
    print(f"- 필터링 제외: {filtered_out}개")
    print(f"- 저장된 이미지: {saved_images}개")

# 실행
input_folder = "./data/aihub/dog/original/validation/normal/yes/A6"  # 원본 데이터 폴더
output_folder = "./data/filtered_images/dog/validation/A6"  # 학습 가능한 이미지 저장 폴더
filter_images(input_folder, output_folder)