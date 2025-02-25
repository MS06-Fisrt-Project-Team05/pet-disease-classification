import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# 데이터 전처리 설정
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# CIFAR-10 데이터셋 로드
train_dataset = datasets.CIFAR10(root='../../data/test', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='../../data/test', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 모델 불러오기 (각 모델별 맞는 weights 설정)
resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
resnet.fc = nn.Linear(resnet.fc.in_features, 10)  # CIFAR-10은 10개의 클래스

efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
efficientnet.classifier[1] = nn.Linear(efficientnet.classifier[1].in_features, 10)

# 손실 함수 및 옵티마이저 설정
criterion = nn.CrossEntropyLoss()
optimizer_resnet = optim.Adam(resnet.parameters(), lr=0.001)
optimizer_efficientnet = optim.Adam(efficientnet.parameters(), lr=0.001)

# 모델 학습 함수 (Loss 출력 추가)
def train(model, optimizer, train_loader, criterion, device, epoch):
    model.train()
    model.to(device)
    running_loss = 0.0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch}] Batch [{batch_idx}] Loss: {loss.item():.4f}")

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch}] Average Loss: {avg_loss:.4f}")

# 모델 평가 함수
def evaluate(model, test_loader, device):
    model.eval()
    model.to(device)
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = 100. * correct / len(test_loader.dataset)
    return accuracy

# 앙상블 예측 함수 (Soft Voting)
def ensemble_evaluate(models, test_loader, device):
    for model in models:
        model.eval()
        model.to(device)

    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = [model(data) for model in models]
            avg_output = sum(outputs) / len(models)  # Soft Voting (평균 확률 계산)
            pred = avg_output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = 100. * correct / len(test_loader.dataset)
    return accuracy

# 학습 및 평가 실행
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ResNet 학습 및 평가
resnet.to(device)
for epoch in range(1, 6):  # 5 에포크 동안 학습
    train(resnet, optimizer_resnet, train_loader, criterion, device, epoch)
resnet_accuracy = evaluate(resnet, test_loader, device)
print(f'ResNet Accuracy: {resnet_accuracy:.2f}%')

# EfficientNet 학습 및 평가
efficientnet.to(device)
for epoch in range(1, 6):
    train(efficientnet, optimizer_efficientnet, train_loader, criterion, device, epoch)
efficientnet_accuracy = evaluate(efficientnet, test_loader, device)
print(f'EfficientNet Accuracy: {efficientnet_accuracy:.2f}%')

# 앙상블 모델 평가
ensemble_accuracy = ensemble_evaluate([resnet, efficientnet], test_loader, device)
print(f'Ensemble Model Accuracy: {ensemble_accuracy:.2f}%')