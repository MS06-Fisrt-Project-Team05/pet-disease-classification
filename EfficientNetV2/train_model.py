import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

import timm  # pip install timm

# ----- 학습용 함수 -----
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    
    # tqdm으로 진행바 추가
    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 통계 계산
        total_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        batch_correct = (preds == labels).sum().item()
        total_correct += batch_correct
        
        # 현재 배치의 정확도 계산
        batch_acc = batch_correct / images.size(0)
        
        # 진행바 업데이트
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'batch_acc': f'{batch_acc:.4f}'
        })

    avg_loss = total_loss / len(train_loader.dataset)
    avg_acc = total_correct / len(train_loader.dataset)
    return avg_loss, avg_acc


# ----- 검증용 함수 -----
def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0

    # 검증용 진행바 추가
    pbar = tqdm(val_loader, desc='Validating')
    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            batch_correct = (preds == labels).sum().item()
            total_correct += batch_correct
            
            # 현재 배치의 정확도 계산
            batch_acc = batch_correct / images.size(0)
            
            # 진행바 업데이트
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'batch_acc': f'{batch_acc:.4f}'
            })

    avg_loss = total_loss / len(val_loader.dataset)
    avg_acc = total_correct / len(val_loader.dataset)
    return avg_loss, avg_acc


def main():
    # ----- 하이퍼파라미터 설정 -----
    batch_size = 32
    lr = 1e-3
    epochs = 10
    num_classes = 6  # 6개 클래스 분류

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ----- 데이터 전처리 파이프라인 -----
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # EfficientNetV2는 224×224 이상 권장
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet 통계값
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # ----- 학습용/검증용 데이터셋 -----
    train_dataset = datasets.ImageFolder(
        root="../../../data/balanced-data/train",
        transform=transform
    )
    val_dataset = datasets.ImageFolder(
        root="../../../data/balanced-data/validation",
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # ----- 모델 불러오기 (timm) -----
    # tf_efficientnetv2_s, tf_efficientnetv2_m, tf_efficientnetv2_l 등 다양한 버전 존재
    model = timm.create_model("tf_efficientnetv2_s", pretrained=True, num_classes=num_classes)
    model.to(device)

    # ----- 손실 함수 & 최적화 기법 설정 -----
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # ----- 학습 루프 -----
    for epoch in range(epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Epoch [{epoch+1}/{epochs}]")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

    # ----- 학습된 모델 저장 -----
    torch.save(model.state_dict(), "efficientnetv2_6class.pth")
    print("Model saved as efficientnetv2_6class.pth")

if __name__ == "__main__":
    main()
