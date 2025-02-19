import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

# =============================================================================
# 1. 커스텀 데이터셋 클래스 (고정 이미지 수와 8:2 샘플링 적용)
# =============================================================================
class FixedSplitSymptomDataset(Dataset):
    def __init__(self, root_dir, symptom_folders, mode="train", train_preprocess_folders=None,
                 fixed_total=100, train_ratio=0.8, transform=None):
        """
        각 증상 폴더마다 고정된 이미지 수(fixed_total) 중,
         - 학습(mode="train")일 때: 지정된 전처리 폴더(예: "color", "normalize", "histogram")에서 랜덤 샘플링하여 (fixed_total * train_ratio) 사용
         - 테스트(mode="test")일 때: "original" 폴더에서 랜덤 샘플링하여 나머지 사용

        Args:
            root_dir (str): 예) "data/dog" 또는 "data/cat"
            symptom_folders (list of str): 사용할 증상 폴더 리스트 (예: ["A1", "A2"] 또는 ["A2", "A4", "A6"] 등)
            mode (str): "train" 또는 "test"
            train_preprocess_folders (list of str): 학습 시 사용할 전처리 폴더 리스트.
            fixed_total (int): 각 증상당 전체 사용 이미지 수 (예: 100)
            train_ratio (float): 학습 이미지 비율 (예: 0.8)
            transform: torchvision.transforms 적용 함수
        """
        self.samples = []
        self.transform = transform
        self.mode = mode
        self.fixed_total = fixed_total
        self.train_ratio = train_ratio
        self.desired_train_count = int(fixed_total * train_ratio)
        self.desired_test_count = fixed_total - self.desired_train_count

        print(f"==== Mode: {self.mode} ====")
        print(f"Desired per symptom: Train = {self.desired_train_count}, Test = {self.desired_test_count}\n")
        
        for label, symptom in enumerate(symptom_folders):
            symptom_path = os.path.join(root_dir, symptom)
            print(f"Processing symptom folder: {symptom} (label {label})")
            if self.mode == "train":
                # 학습 시: 지정된 전처리 폴더에서 이미지 수집
                if train_preprocess_folders is None:
                    # "original" 폴더를 제외한 모든 폴더 사용
                    available_folders = [d for d in os.listdir(symptom_path)
                                         if os.path.isdir(os.path.join(symptom_path, d)) and d != "original"]
                else:
                    available_folders = train_preprocess_folders
                proc_images = []
                for folder in available_folders:
                    folder_path = os.path.join(symptom_path, folder)
                    if os.path.isdir(folder_path):
                        files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                        proc_images.extend(files)
                        print(f"  Found {len(files)} images in '{folder}'")
                    else:
                        print(f"  Folder '{folder}' not found!")
                print(f"  Total preprocessed images found: {len(proc_images)}")
                if len(proc_images) >= self.desired_train_count:
                    selected = random.sample(proc_images, self.desired_train_count)
                else:
                    print(f"  Warning: Only {len(proc_images)} images available (desired {self.desired_train_count}).")
                    selected = proc_images
                for img_path in selected:
                    self.samples.append((img_path, label))
                print(f"  -> Using {len(selected)} images for training.\n")
            else:  # mode == "test"
                # 테스트 시: "original" 폴더에서 이미지 수집
                orig_folder = os.path.join(symptom_path, "original")
                if os.path.isdir(orig_folder):
                    orig_images = [os.path.join(orig_folder, f) for f in os.listdir(orig_folder)
                                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    print(f"  Found {len(orig_images)} original images in 'original'")
                    if len(orig_images) >= self.desired_test_count:
                        selected = random.sample(orig_images, self.desired_test_count)
                    else:
                        print(f"  Warning: Only {len(orig_images)} original images available (desired {self.desired_test_count}).")
                        selected = orig_images
                    for img_path in selected:
                        self.samples.append((img_path, label))
                    print(f"  -> Using {len(selected)} images for testing.\n")
                else:
                    print(f"  Warning: 'original' folder not found for symptom {symptom}!\n")
        
        print(f"Total samples in {self.mode} dataset: {len(self.samples)}\n")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# =============================================================================
# 2. DataLoader 생성 함수 (데이터셋이 비어있으면 에러 메시지 출력)
# =============================================================================
def create_dataloader(root_dir, symptom_folders, mode, train_preprocess_folders=None,
                      fixed_total=100, train_ratio=0.8, transform=None, batch_size=32, shuffle=True):
    dataset = FixedSplitSymptomDataset(root_dir, symptom_folders, mode,
                                        train_preprocess_folders, fixed_total, train_ratio, transform)
    if len(dataset) == 0:
        err_msg = f"Dataset is empty for mode '{mode}' with symptoms {symptom_folders} and preprocess folders {train_preprocess_folders}"
        print(err_msg)
        raise ValueError(err_msg)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    print(f"Created DataLoader with {len(dataset)} samples for mode '{mode}'\n")
    return dataloader

# =============================================================================
# 3. 학습/평가를 위한 더미 학습 루프 (디버그용 print 포함)
# =============================================================================
def dummy_train(model, dataloader, optimizer, criterion, device, epoch):
    model.train()
    running_loss = 0.0
    total = 0
    correct = 0
    for i, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
        
        if i % 5 == 0:
            print(f"Epoch {epoch}, Batch {i}: Loss = {loss.item():.4f}")
    epoch_loss = running_loss / len(dataloader)
    accuracy = correct / total * 100
    print(f"Epoch {epoch} finished: Loss = {epoch_loss:.4f}, Accuracy = {accuracy:.2f}%\n")
    return model

# =============================================================================
# 4. 메인 함수: 증상 그룹별(점진적) 학습 실행 예제
# =============================================================================
def main():
    # 재현성을 위한 시드 고정
    random.seed(42)
    
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # pet_type 설정: "dog" 또는 "cat"
    pet_type = "dog"  # 테스트 시 원하는 값으로 변경 가능
    print(f"Selected pet type: {pet_type}\n")
    
    # 이미지 변환 설정 (예시)
    train_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
    ])
    test_transform = train_transform
    
    # 현재 파일이 hunchan/pet-disease-classification/ResNet 아래 있으므로,
    # 데이터 폴더는 상위 3단계에 위치합니다.
    root_dir = os.path.join(os.path.dirname(__file__), "../../../data", pet_type)
    print(f"Root directory: {root_dir}\n")
    
    # pet_type에 따른 증상 그룹 구성
    if pet_type == "dog":
        incremental_steps = [
            {
                "name": "Step 1: A1 + A2",
                "symptoms": ["A1", "A2"],
                "train_preprocess": ["color", "normalize"]  # Step 1: "color", "normalize" 사용
            },
            {
                "name": "Step 2: A3 + A4",
                "symptoms": ["A3", "A4"],
                "train_preprocess": ["histogram"]  # Step 2: "histogram" 사용
            },
            {
                "name": "Step 3: A5 + A6",
                "symptoms": ["A5", "A6"],
                "train_preprocess": ["color", "normalize"]  # Step 3: "color", "normalize" 사용
            }
        ]
    else:  # pet_type == "cat"
        incremental_steps = [
            {
                "name": "Step 1: A2",
                "symptoms": ["A2"],
                "train_preprocess": ["color", "normalize"]
            },
            {
                "name": "Step 2: A4",
                "symptoms": ["A4"],
                "train_preprocess": ["histogram"]
            },
            {
                "name": "Step 3: A6",
                "symptoms": ["A6"],
                "train_preprocess": ["color", "normalize"]
            }
        ]
    
    print("Incremental steps:")
    for step in incremental_steps:
        print(f"  {step['name']} -> Symptoms: {step['symptoms']}, Preprocess: {step['train_preprocess']}")
    print("")
    
    # 더미 모델 생성 (실제 모델로 교체 가능)
    # (입력 이미지 크기는 224x224x3, 출력 클래스 수는 pet_type에 따라 다르게 지정 가능)
    if pet_type == "dog":
        num_classes = 6
    else:
        num_classes = 3
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(224*224*3, num_classes)
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 각 증상 그룹별로 점진적 학습 수행
    for step in incremental_steps:
        print(f"\n=== {step['name']} ===")
        try:
            # 학습 및 테스트 DataLoader 생성
            train_loader = create_dataloader(root_dir, step["symptoms"], mode="train",
                                             train_preprocess_folders=step["train_preprocess"],
                                             fixed_total=100, train_ratio=0.8,
                                             transform=train_transform, batch_size=16, shuffle=True)
            test_loader = create_dataloader(root_dir, step["symptoms"], mode="test",
                                            fixed_total=100, train_ratio=0.8,
                                            transform=test_transform, batch_size=16, shuffle=False)
        except ValueError as e:
            print(f"Skipping {step['name']} due to error: {e}")
            continue
        
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Testing samples: {len(test_loader.dataset)}\n")
        
        # 각 단계마다 2 epoch 학습 (예시)
        for epoch in range(1, 3):
            print(f"Starting epoch {epoch} for {step['name']}")
            model = dummy_train(model, train_loader, optimizer, criterion, device, epoch)
        print(f"Finished {step['name']}\n")
    
    # 최종 모델 저장
    torch.save(model.state_dict(), "final_pet_disease_model.pth")
    print("Final model saved as final_pet_disease_model.pth")

if __name__ == "__main__":
    main()
