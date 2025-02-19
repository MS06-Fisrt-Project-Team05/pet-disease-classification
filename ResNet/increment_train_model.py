import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Subset
from datetime import datetime
import json

# Device 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

# 공통 이미지 변환 설정
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
}

# 분할된 데이터 경로 (data_splitter.py로 생성한 데이터를 사용)
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data/selected/dog"))
train_dir = os.path.join(data_dir, 'train')
val_dir   = os.path.join(data_dir, 'val')

print(f"Train directory: {train_dir}")
print(f"Validation directory: {val_dir}\n")

# 전체 데이터셋 (ImageFolder)
full_train_dataset = datasets.ImageFolder(train_dir, data_transforms['train'])
full_val_dataset   = datasets.ImageFolder(val_dir, data_transforms['val'])
class_names = full_train_dataset.classes
num_classes = len(class_names)
print("Classes:", class_names, "\n")

# --- Step별로 사용할 클래스 그룹 ---
# 예: dog의 경우
step1_classes = ['A1', 'A2']
step2_classes = ['A3', 'A4']
step3_classes = ['A5', 'A6']

def get_subset_indices(dataset, target_classes):
    indices = [i for i, (_, label) in enumerate(dataset.samples)
               if dataset.classes[label] in target_classes]
    return indices

# Step 1: A1, A2
step1_train_idx = get_subset_indices(full_train_dataset, step1_classes)
step1_val_idx   = get_subset_indices(full_val_dataset, step1_classes)
step1_train_ds  = Subset(full_train_dataset, step1_train_idx)
step1_val_ds    = Subset(full_val_dataset, step1_val_idx)

# Step 2: A3, A4
step2_train_idx = get_subset_indices(full_train_dataset, step2_classes)
step2_val_idx   = get_subset_indices(full_val_dataset, step2_classes)
step2_train_ds  = Subset(full_train_dataset, step2_train_idx)
step2_val_ds    = Subset(full_val_dataset, step2_val_idx)

# Step 3: A5, A6
step3_train_idx = get_subset_indices(full_train_dataset, step3_classes)
step3_val_idx   = get_subset_indices(full_val_dataset, step3_classes)
step3_train_ds  = Subset(full_train_dataset, step3_train_idx)
step3_val_ds    = Subset(full_val_dataset, step3_val_idx)

def create_loader(dataset, batch_size=32, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)

# DataLoader 생성
train_loader_step1 = create_loader(step1_train_ds)
val_loader_step1   = create_loader(step1_val_ds, shuffle=False)
train_loader_step2 = create_loader(step2_train_ds)
val_loader_step2   = create_loader(step2_val_ds, shuffle=False)
train_loader_step3 = create_loader(step3_train_ds)
val_loader_step3   = create_loader(step3_val_ds, shuffle=False)

print("Dataset sizes per step:")
print(f"  Step 1: Train = {len(step1_train_ds)}, Val = {len(step1_val_ds)}")
print(f"  Step 2: Train = {len(step2_train_ds)}, Val = {len(step2_val_ds)}")
print(f"  Step 3: Train = {len(step3_train_ds)}, Val = {len(step3_val_ds)}\n")

# 모델 구성 (ResNet50; 출력 클래스는 전체 6개로 설정)
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

def train_step(model, train_loader, val_loader, num_epochs, step_name):
    print(f"\n--- {step_name} 시작 ---")
    best_model_wts = model.state_dict()
    best_acc = 0.0
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs} [{step_name}]")
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc  = running_corrects.double() / len(train_loader.dataset)
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
        val_loss = running_loss / len(val_loader.dataset)
        val_acc  = running_corrects.double() / len(val_loader.dataset)
        print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = model.state_dict()
            print("새로운 최고 정확도 달성!")
        
        scheduler.step()
    model.load_state_dict(best_model_wts)
    print(f"--- {step_name} 완료, Best Val Acc: {best_acc:.4f} ---\n")
    return model

# Step 1 학습: A1 + A2
model = train_step(model, train_loader_step1, val_loader_step1, num_epochs=2, step_name="Step 1: A1+A2")
torch.save(model.state_dict(), "model_step1.pth")
print("Step 1 모델 저장: model_step1.pth\n")

# Step 2 학습: A3 + A4 (이전 단계 모델 불러오기)
model.load_state_dict(torch.load("model_step1.pth"))
model = train_step(model, train_loader_step2, val_loader_step2, num_epochs=2, step_name="Step 2: A3+A4")
torch.save(model.state_dict(), "model_step2.pth")
print("Step 2 모델 저장: model_step2.pth\n")

# Step 3 학습: A5 + A6 (이전 단계 모델 불러오기)
model.load_state_dict(torch.load("model_step2.pth"))
model = train_step(model, train_loader_step3, val_loader_step3, num_epochs=2, step_name="Step 3: A5+A6")
torch.save(model.state_dict(), "model_step3.pth")
print("Step 3 모델 저장: model_step3.pth\n")

# 클래스 정보를 json 파일로 저장 (optional)
base_class_names_path = "class_names.json"
if os.path.exists(base_class_names_path):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    class_names_path = f"class_names_{timestamp}.json"
else:
    class_names_path = base_class_names_path

with open(class_names_path, 'w', encoding='utf-8') as f:
    json.dump(class_names, f, ensure_ascii=False, indent=4)
print(f"Class names saved to {class_names_path}")
