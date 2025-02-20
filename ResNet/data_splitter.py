import os
import shutil
import random

def split_data(source_dir, dest_dir, pet_type, fixed_total=100, train_ratio=0.8):
    """
    각 클래스별로 source_dir의 데이터를 고정 수량(fixed_total)만큼 선택하여,
    train은 지정된 전처리 폴더에서, val은 "original" 폴더에서 각각 랜덤 추출한 후
    dest_dir에 아래 구조로 복사합니다.
    
    dest_dir/
        train/<class>/
        val/<class>/
    
    [dog의 경우]
      - A1, A2, A5, A6: train 이미지는 ["color", "normalize"]
      - A3, A4: train 이미지는 ["histogram"]
      - val 이미지는 각 클래스의 "original" 폴더에서 선택

    [cat의 경우]
      - A2, A6: train 이미지는 ["color", "normalize"]
      - A4: train 이미지는 ["histogram"]
      - val 이미지는 각 클래스의 "original" 폴더에서 선택
    """
    if pet_type.lower() == "dog":
        train_mapping = {
            "A1": ["normalize", "histogram", "color"],
            "A2": [ "normalize", "histogram", "color"],
            "A3": [ "color", "normalize", "histogram"],
            "A4": ["color", "normalize", "histogram"],
            "A5": ["color", "normalize", "histogram"],
            "A6": ["color", "normalize", "histogram"],
        }
    elif pet_type.lower() == "cat":
        train_mapping = {
            "A2": ["color", "normalize"],
            "A4": ["histogram"],
            "A6": ["color", "normalize"],
        }
    else:
        raise ValueError("pet_type must be 'dog' or 'cat'")

    classes = list(train_mapping.keys())
    desired_train_count = int(fixed_total * train_ratio)
    desired_val_count = fixed_total - desired_train_count

    print(f"Splitting data for pet type: {pet_type}")
    print(f"Each class: total {fixed_total} images, train: {desired_train_count}, val: {desired_val_count}")
    print(f"Classes to process: {classes}\n")

    # destination 디렉터리 생성
    train_dest = os.path.join(dest_dir, "train")
    val_dest = os.path.join(dest_dir, "val")
    os.makedirs(train_dest, exist_ok=True)
    os.makedirs(val_dest, exist_ok=True)

    for cls in classes:
        print(f"Processing class {cls}...")
        source_cls_dir = os.path.join(source_dir, cls)
        # train 이미지: 지정된 전처리 폴더에서 모두 모으기
        preprocess_folders = train_mapping[cls]
        train_images = []
        for folder in preprocess_folders:
            folder_path = os.path.join(source_cls_dir, folder)
            if os.path.isdir(folder_path):
                imgs = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                        if f.lower().endswith(('.png','.jpg','.jpeg'))]
                train_images.extend(imgs)
                print(f"  Found {len(imgs)} images in folder '{folder}'")
            else:
                print(f"  Warning: Folder '{folder}' not found in class {cls}")
        print(f"  Total training images found: {len(train_images)}")
        if len(train_images) < desired_train_count:
            print(f"  Warning: Only {len(train_images)} training images available, desired {desired_train_count}.")
            selected_train = train_images
        else:
            selected_train = random.sample(train_images, desired_train_count)

        # val 이미지: "original" 폴더에서 선택
        original_folder = os.path.join(source_cls_dir, "original")
        if os.path.isdir(original_folder):
            val_images = [os.path.join(original_folder, f) for f in os.listdir(original_folder)
                          if f.lower().endswith(('.png','.jpg','.jpeg'))]
            print(f"  Found {len(val_images)} images in 'original' folder")
        else:
            print(f"  Warning: 'original' folder not found in class {cls}")
            val_images = []
        if len(val_images) < desired_val_count:
            print(f"  Warning: Only {len(val_images)} validation images available, desired {desired_val_count}.")
            selected_val = val_images
        else:
            selected_val = random.sample(val_images, desired_val_count)

        # destination 경로 생성
        dest_train_cls = os.path.join(train_dest, cls)
        dest_val_cls = os.path.join(val_dest, cls)
        os.makedirs(dest_train_cls, exist_ok=True)
        os.makedirs(dest_val_cls, exist_ok=True)

        # 파일 복사
        for img_path in selected_train:
            shutil.copy(img_path, dest_train_cls)
        for img_path in selected_val:
            shutil.copy(img_path, dest_val_cls)

        print(f"  Copied {len(selected_train)} training images to {dest_train_cls}")
        print(f"  Copied {len(selected_val)} validation images to {dest_val_cls}\n")

    print("Data splitting and copying complete.\n")

if __name__ == '__main__':
    # 직접 실행할 경우 예제 사용:
    source_dir = '../../../data/cat'
    dest_dir = '../../../data/selected/cat'
    pet_type = "cat"
    split_data(source_dir, dest_dir, pet_type, fixed_total=300, train_ratio=0.8)
