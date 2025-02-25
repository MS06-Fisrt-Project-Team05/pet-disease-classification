import os
import json
import copy
import argparse
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models

# =====================================
# 1. 파라미터 설정
# =====================================
def parse_args():
    parser = argparse.ArgumentParser(description="Train Skin Disease Classification Model")
    parser.add_argument("--model", type=str, default="resnet50", 
                        choices=["resnet18", "resnet50", 
                                 "efficientnet_b0", "efficientnet_b4", 
                                 "ensemble"])
    parser.add_argument("--animal", type=str, default="dog", choices=["dog", "cat"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--freeze_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--mixup_alpha", type=float, default=0.3)
    parser.add_argument("--aug_strength", type=float, default=0.5)
    return parser.parse_args()

# =====================================
# 2. 데이터 로더
# =====================================
def get_dataloaders(animal_type, batch_size, aug_strength=0.5):
    base_path = f"../../data/good-data/{animal_type}/selected"
    train_dir = os.path.join(base_path, "train")
    val_dir = os.path.join(base_path, "validation")

    # 데이터 증강
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7+aug_strength*0.1, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5+aug_strength*0.1),
        transforms.ColorJitter(brightness=0.2+aug_strength*0.1, contrast=0.2+aug_strength*0.1),
        transforms.RandomRotation(30),
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15)),
        transforms.GaussianBlur(3, sigma=(0.1, 0.5)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 클래스 밸런스 조정 (WeightedRandomSampler)
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    class_counts = np.array([len([x for x in train_dataset.samples if x[1] == cls]) 
                             for cls in range(len(train_dataset.classes))])
    class_weights = 1. / class_counts
    sample_weights = class_weights[train_dataset.targets]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    # 데이터 로더
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4)
    val_loader = DataLoader(datasets.ImageFolder(val_dir, transform=val_transform), 
                           batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, train_dataset.classes

# =====================================
# 3. 모델 구성
# =====================================
def build_model(model_name, num_classes):
    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Sequential(
            nn.Dropout(0.7),
            nn.Linear(model.fc.in_features, num_classes)
        )
    elif model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Sequential(
            nn.Dropout(0.7),
            nn.Linear(model.fc.in_features, num_classes)
        )
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Sequential(
            nn.Dropout(0.7),
            nn.Linear(model.classifier[1].in_features, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    elif model_name == "efficientnet_b4":
        model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
        # B4는 classifier 구조가 [Dropout, Linear]이므로 classifier[1]이 최종 Linear
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Dropout(0.7),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    return model

# =====================================
# 4. 학습 함수
# =====================================
def mixup_data(x, y, alpha=0.3):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def train_epoch(model, loader, criterion, optimizer, device, mixup_alpha):
    model.train()
    total_loss, correct = 0, 0
    
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        if mixup_alpha > 0:
            inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, mixup_alpha)
            outputs = model(inputs)
            loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * inputs.size(0)
        with torch.no_grad():
            preds = outputs.argmax(dim=1)
            if mixup_alpha > 0:
                correct += (lam * preds.eq(labels_a).sum().item() 
                           + (1 - lam) * preds.eq(labels_b).sum().item())
            else:
                correct += preds.eq(labels).sum().item()
    
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0, 0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct += preds.eq(labels).sum().item()
    
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

# =====================================
# 5. 앙상블 관련 함수 (소프트 보팅)
# =====================================
class EnsembleModel(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)
        
    def forward(self, x):
        # 각각의 모델 출력(logits)을 평균 -> 소프트 보팅
        outputs = [model(x) for model in self.models]
        return torch.mean(torch.stack(outputs), dim=0)

def train_ensemble(models, train_loaders, val_loader, num_classes, device, args):
    best_acc = 0.0
    best_models = []  # 최고 성능 모델 저장용
    optimizers = [optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) for model in models]
    
    print("\n=== Ensemble Training Configuration ===")
    print(f"Total Models: {len(models)}")
    print(f"Model Types: {[type(m).__name__ for m in models]}")
    print(f"Individual Learning Rate: {args.lr}")
    print(f"Mixup Alpha: {args.mixup_alpha}\n")

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 각 모델별 학습
        for i, (model, optimizer) in enumerate(zip(models, optimizers)):
            model.train()
            print(f"\nTraining Model {i+1}")
            train_loss, train_acc = train_epoch(model, train_loaders[i], nn.CrossEntropyLoss(), 
                                                optimizer, device, args.mixup_alpha)
            print(f"Model {i+1} - Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        
        # 앙상블 검증
        ensemble = EnsembleModel(models)
        val_loss, val_acc = validate(ensemble, val_loader, nn.CrossEntropyLoss(), device)
        print(f"\n[Ensemble Evaluation] Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")
        
        # 최고 성능 모델 갱신 시 저장
        if val_acc > best_acc:
            best_acc = val_acc
            best_models = [copy.deepcopy(model.state_dict()) for model in models]
            print(f"New best ensemble found at epoch {epoch+1} with acc: {best_acc:.4f}")

    # 최종 모델 로드 및 저장
    for i, model in enumerate(models):
        model.load_state_dict(best_models[i])
    final_ensemble = EnsembleModel(models)
    final_loss, final_acc = validate(final_ensemble, val_loader, nn.CrossEntropyLoss(), device)
    
    print(f"\n=== Final Ensemble Evaluation ===")
    print(f"Best Ensemble Accuracy: {final_acc:.4f}")
    
    save_time = datetime.now().strftime("%m%d_%H%M")
    ensemble_save_name = f"{args.animal}_full_ensemble_{save_time}_acc{final_acc:.4f}.pth"
    torch.save({
        'model1': best_models[0],
        'model2': best_models[1],
    }, ensemble_save_name)
    print(f"Ensemble model saved as {ensemble_save_name}.")

# =====================================
# 6. 메인 함수
# =====================================
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n=== Training Configuration ===")
    print(f"Animal Type: {args.animal.upper()}")
    print(f"Model Type: {args.model.upper()}")
    print(f"Total Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Initial LR: {args.lr}")
    print(f"Mixup Alpha: {args.mixup_alpha}")
    print(f"Augmentation Strength: {args.aug_strength}\n")
    
    # 데이터 로드
    train_loader, val_loader, class_names = get_dataloaders(args.animal, args.batch_size, args.aug_strength)
    num_classes = len(class_names)
    
    print("\n=== Class Information ===")
    print(f"Total Classes: {num_classes}")
    print(f"Class Names: {class_names}\n")
    
    # 단일 모델 학습
    if args.model != "ensemble":
        model = build_model(args.model, num_classes).to(device)
        
        print("\n=== Model Architecture ===")
        print(model)
        print(f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")
        
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
        
        best_acc = 0.0
        best_state = None
        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            print(f"Current LR: {optimizer.param_groups[0]['lr']:.2e}")
            
            train_loss, train_acc = train_epoch(model, train_loader, nn.CrossEntropyLoss(), 
                                                optimizer, device, args.mixup_alpha)
            val_loss, val_acc = validate(model, val_loader, nn.CrossEntropyLoss(), device)
            scheduler.step()
            
            print(f"\n[Validation] Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")
            
            if val_acc > best_acc:
                best_acc = val_acc
                best_state = copy.deepcopy(model.state_dict())
                save_time = datetime.now().strftime("%m%d_%H%M")
                save_name = f"{args.animal}_best_{args.model}_{save_time}_acc{val_acc:.4f}.pth"
                torch.save(best_state, save_name)
                print(f"Saved new best model: {save_name}")
                
        # 최종 결과 출력
        if best_state is not None:
            model.load_state_dict(best_state)
        print(f"\n=== Final Best Accuracy: {best_acc:.4f} ===")
        
    # 앙상블 학습 (ResNet50 + EfficientNet-B4 예시)
    else:
        model1 = build_model('resnet50', num_classes)
        model2 = build_model('efficientnet_b4', num_classes)
        
        # 모델별로 다른 증강 강도를 시도할 수 있음 (예: model1=0.5, model2=0.7)
        train_loader1, _, _ = get_dataloaders(args.animal, args.batch_size, aug_strength=0.5)
        train_loader2, _, _ = get_dataloaders(args.animal, args.batch_size, aug_strength=0.7)
        
        models = [model1.to(device), model2.to(device)]
        train_ensemble(models, [train_loader1, train_loader2], val_loader, num_classes, device, args)

if __name__ == "__main__":
    main()
