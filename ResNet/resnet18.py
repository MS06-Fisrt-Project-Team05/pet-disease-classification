import os
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import json

# ====== 데이터 경로 및 전처리 (Data Augmentation 적용) ======
data_dir = '../../../data/balanced-data'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'validation')

print(f"Train directory: {train_dir}")
print(f"Validation directory: {val_dir}\n")

# 데이터 증강 (더 다양한 augmentation 추가)
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),            # 랜덤 크롭
        transforms.RandomHorizontalFlip(),            # 좌우 반전
        transforms.RandomVerticalFlip(),              # 상하 반전
        transforms.RandomRotation(15),                # 15도 범위 내 회전
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 색상 변화
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
}

image_datasets = {
    'train': datasets.ImageFolder(train_dir, data_transforms['train']),
    'val': datasets.ImageFolder(val_dir, data_transforms['val'])
}

dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=32, shuffle=True, num_workers=4),
    'val': DataLoader(image_datasets['val'], batch_size=32, shuffle=False, num_workers=4)
}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
print("Dataset sizes:", dataset_sizes)  # train: 2712, val: 678 (예상)

class_names = image_datasets['train'].classes
num_classes = len(class_names)

# 클래스 정보를 json 파일로 저장 (모델 저장 시 사용)
base_class_names_path = "class_names.json"
if os.path.exists(base_class_names_path):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    class_names_path = f"class_names_{timestamp}.json"
else:
    class_names_path = base_class_names_path

with open(class_names_path, 'w', encoding='utf-8') as f:
    json.dump(class_names, f, ensure_ascii=False, indent=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

# ====== 모델 구성 및 Fine-Tuning 전략 ======
# 사전 학습된 ResNet18 사용 (ResNet18으로 변경)
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_features = model.fc.in_features

# 분류기(fc)에 Dropout 추가 (과적합 방지)
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_features, num_classes)
)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

def train_model(model, criterion, optimizer, scheduler, num_epochs=25, freeze_epochs=5):
    print("\n====== 학습 시작 ======")
    print(f"총 에포크 수: {num_epochs}")
    print(f"Feature extractor 고정 에포크 수: {freeze_epochs}")
    print(f"학습 데이터 크기: {dataset_sizes['train']}")
    print(f"검증 데이터 크기: {dataset_sizes['val']}")
    print("=" * 50)

    best_model_wts = model.state_dict()
    best_acc = 0.0

    def set_parameter_requires_grad(model, requires_grad):
        for param in model.parameters():
            param.requires_grad = requires_grad

    for epoch in range(num_epochs):
        print(f'\n{"="*20} Epoch {epoch + 1}/{num_epochs} {"="*20}')

        # 초기 freeze_epochs 동안 feature extractor 동결
        if epoch < freeze_epochs:
            print("\n[학습 설정] Feature extractor 고정 중...")
            set_parameter_requires_grad(model, False)
            for param in model.fc.parameters():
                param.requires_grad = True
        else:
            print("\n[학습 설정] 모든 레이어 미세조정 중...")
            set_parameter_requires_grad(model, True)

        for phase in ['train', 'val']:
            if phase == 'train':
                print(f'\n[Phase] 학습 단계 시작')
                model.train()
            else:
                print(f'\n[Phase] 검증 단계 시작')
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            total_batches = len(dataloaders[phase])

            for batch_idx, (inputs, labels) in enumerate(dataloaders[phase]):
                if batch_idx % 5 == 0:  # 더 자주 진행상황 출력
                    print(f'진행률: {batch_idx}/{total_batches} 배치 처리 완료 ({(100. * batch_idx / total_batches):.1f}%) | ', end='')
                    if running_corrects > 0 and batch_idx > 0:
                        current_acc = running_corrects.double() / ((batch_idx+1) * inputs.size(0))
                        print(f'현재 정확도: {current_acc:.4f}')
                    else:
                        print()

                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'\n[{phase.upper()} 결과]')
            print(f'손실: {epoch_loss:.4f} | 정확도: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                print(f'=> 새로운 최고 정확도 달성! ({epoch_acc:.4f})')

        # 현재 학습률 출력
        current_lr = optimizer.param_groups[0]['lr']
        print(f'\n현재 학습률: {current_lr}')
        print("=" * 60)

    print('\n====== 학습 완료 ======')
    print(f'최종 최고 검증 정확도: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    return model

num_epochs = 25
model = train_model(model, criterion, optimizer, scheduler, num_epochs=num_epochs, freeze_epochs=5)

torch.save(model.state_dict(), 'resnet18_cat_disease.pth')
print("Model saved to resnet18_cat_disease.pth")
