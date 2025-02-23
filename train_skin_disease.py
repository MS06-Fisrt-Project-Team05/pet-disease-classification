import os
import json
import copy
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from torchvision import datasets, transforms, models

# =====================================
# 1. 파라미터 & 경로 설정
# =====================================
def parse_args():
    parser = argparse.ArgumentParser(description="Train Skin Disease Classification Model")
    parser.add_argument("--model", type=str, default="resnet",
                        choices=["resnet", "efficientnet"],
                        help="Select model architecture: 'resnet' or 'efficientnet'")
    parser.add_argument("--animal", type=str, default="dog",
                        choices=["dog", "cat"],
                        help="Choose which animal: 'dog' or 'cat'")
    parser.add_argument("--epochs", type=int, default=25,
                        help="Number of epochs to train (default: 25)")
    parser.add_argument("--freeze_epochs", type=int, default=5,
                        help="Number of initial epochs to freeze backbone (default: 5)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default: 1e-3)")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay (default: 1e-4)")
    args = parser.parse_args()
    return args


# =====================================
# 2. 모델 빌드
# =====================================
def build_model(model_name, num_classes, pretrained=True):
    """
    model_name: "resnet" | "efficientnet"
    num_classes: 7 (A1, A2, A3, A4, A5, A6, Negative)
    """
    if model_name == "resnet":
        # 사전 학습된 resnet18 or 50 등 원하는 버전을 사용 가능
        # 여기서는 resnet50을 예시로 사용
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        in_features = model.fc.in_features
        # 최종 레이어 교체 (Dropout + Linear)
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes)
        )

    elif model_name == "efficientnet":
        # pip install torchvision>=0.13 이면 torchvision.models.efficientnet도 가능
        # 또는 timm 라이브러리를 사용 가능 (timm.create_model("efficientnet_b0", pretrained=True, num_classes=...))
        # 여기서는 torchvision의 efficientnet_b0 예시
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        in_features = model.classifier[1].in_features
        # 최종 레이어 교체
        model.classifier[1] = nn.Linear(in_features, num_classes)

    else:
        raise ValueError("Unknown model name. Choose 'resnet' or 'efficientnet'.")

    return model


# =====================================
# 3. 학습 함수 / 평가 함수
# =====================================
def set_parameter_requires_grad(model, requires_grad: bool):
    for param in model.parameters():
        param.requires_grad = requires_grad

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_correct = 0
    total_batches = len(dataloader)

    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        running_correct += torch.sum(preds == labels).item()

        # 100배치마다 진행 상황 출력
        if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == total_batches:
            print(f"  Batch {batch_idx+1}/{total_batches} - Loss: {loss.item():.4f}")

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_correct / len(dataloader.dataset)
    return epoch_loss, epoch_acc

def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total_batches = len(dataloader)

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_correct += torch.sum(preds == labels).item()

            # 50배치마다 진행 상황 출력
            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == total_batches:
                print(f"  Val Batch {batch_idx+1}/{total_batches} - Loss: {loss.item():.4f}")

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_correct / len(dataloader.dataset)
    return epoch_loss, epoch_acc

def train_model(model, 
                dataloaders, 
                dataset_sizes, 
                criterion, 
                optimizer, 
                scheduler, 
                device,
                num_epochs=25, 
                freeze_epochs=5):
    """
    freeze_epochs: 초반 freeze_epochs 동안 백본(특징 추출부) 동결, 이후 전체 Unfreeze
    """
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        print(f"Current LR: {scheduler.get_last_lr()[0]:.6f}")  # 학습률 출력

        # 1) Freeze backbone for the first 'freeze_epochs' epochs
        if epoch < freeze_epochs:
            # 백본만 동결, 최종 레이어(fc/classifier)는 학습
            set_parameter_requires_grad(model, False)
            if hasattr(model, "fc"):
                for param in model.fc.parameters():
                    param.requires_grad = True
            elif hasattr(model, "classifier"):
                for param in model.classifier.parameters():
                    param.requires_grad = True
            print(">> [FROZEN] backbone. Only final layer is trainable.")
        else:
            # 2) Unfreeze all layers
            set_parameter_requires_grad(model, True)
            print(">> [UNFROZEN] entire network fine-tuning.")

        # ---------- TRAIN ----------
        print("\nTraining Phase:")
        train_loss, train_acc = train_one_epoch(model, dataloaders["train"], criterion, optimizer, device)
        scheduler.step()

        # ---------- VALIDATE ----------
        print("\nValidation Phase:")
        val_loss, val_acc = validate_one_epoch(model, dataloaders["val"], criterion, device)

        # 정확도 출력 포맷 개선 (소수점 6자리까지)
        print(f"\nTrain Loss: {train_loss:.6f} | Train Acc: {train_acc:.6f}")
        print(f"Val   Loss: {val_loss:.6f} | Val   Acc: {val_acc:.6f}")

        # 최고 정확도 갱신 시 Best Weight 갱신
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            print(f"** Best accuracy updated: {best_acc:.4f}")

    print("\nTraining completed!")
    print(f"Best validation accuracy: {best_acc:.4f}")

    # Best model load
    model.load_state_dict(best_model_wts)
    return model


# =====================================
# 4. 메인 함수: 데이터셋 준비 & 학습 실행
# =====================================
def main():
    args = parse_args()
    model_name = args.model
    animal_type = args.animal
    num_epochs = args.epochs
    freeze_epochs = args.freeze_epochs
    batch_size = args.batch_size
    lr = args.lr
    weight_decay = args.weight_decay

    # ----- 데이터 경로 -----
    # 예) dog: ../../../data/dog/train, ../../../data/dog/validation
    #     cat: ../../../data/cat/train, ../../../data/cat/validation
    # base_path = f"../../data/aihub/{animal_type}"
    base_path = f"../../data/good-data/{animal_type}/selected"
    train_dir = os.path.join(base_path, "train")
    val_dir = os.path.join(base_path, "validation")

    # ----- 데이터셋 준비 -----
    # (A) 학습용 증강 (Data Augmentation)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

    # (B) 검증용 변환 (기본 Resize/CenterCrop만)
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

    image_datasets = {
        "train": datasets.ImageFolder(train_dir, transform=train_transform),
        "val": datasets.ImageFolder(val_dir, transform=val_transform)
    }

    dataloaders = {
        "train": DataLoader(image_datasets["train"], batch_size=batch_size, shuffle=True, num_workers=4),
        "val": DataLoader(image_datasets["val"], batch_size=batch_size, shuffle=False, num_workers=4)
    }

    dataset_sizes = {
        "train": len(image_datasets["train"]),
        "val": len(image_datasets["val"])
    }

    class_names = image_datasets["train"].classes
    num_classes = len(class_names)  # 7 (A1, A2, A3, A4, A5, A6, Negative)

    print(f"Data for [{animal_type}] loaded!")
    print(f"Train samples: {dataset_sizes['train']}, Val samples: {dataset_sizes['val']}")
    print(f"Classes ({num_classes}): {class_names}\n")

    # 클래스 이름 JSON으로 저장
    now_str = datetime.now().strftime("%Y%m%d%H%M%S")
    class_names_path = f"class_names_{animal_type}_{model_name}.json"
    with open(class_names_path, "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=4)

    # ----- 모델 빌드 -----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name, num_classes=num_classes, pretrained=True)
    model.to(device)

    # ----- 최적화 & 스케줄러 -----
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # 7에폭마다 lr 1/10로 감소 (예시)
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

    # ----- 학습 실행 -----
    best_model = train_model(
        model=model,
        dataloaders=dataloaders,
        dataset_sizes=dataset_sizes,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=num_epochs,
        freeze_epochs=freeze_epochs
    )

    # ----- 모델 저장 -----
    # 모델 저장 이름에 타임스탬프 추가
    model_save_name = f"{model_name}_{animal_type}_best_{now_str}.pth"
    torch.save(best_model.state_dict(), model_save_name)
    print(f"\nBest model saved to: {model_save_name}")


if __name__ == "__main__":
    main()
