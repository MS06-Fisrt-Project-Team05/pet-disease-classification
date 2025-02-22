import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from timm import create_model
from PIL import Image
from PIL import ImageFile
from tqdm import tqdm
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast  # Mixed Precision Training

# === 새 데이터 경로 설정 ===
# 예시: 기존 balanced-data에서 selected/cat으로 변경 (학습: 2712개, 검증: 678개)
train_dir = '../../../data/balanced-data/train'
val_dir = '../../../data/balanced-data/validation'
batch_size = 128
num_epochs = 10 
num_classes = 6  # 필요에 따라 분류할 클래스 수 수정 (예: 6 → 원하는 숫자)
early_stopping_patience = 3  # Early Stopping patience
accumulation_steps = 4  # Gradient Accumulation 스텝 수

checkpoint_dir = './checkpoints/'
checkpoint_path = os.path.join(checkpoint_dir, 'val_dog_best_checkpoint.pth')
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === 데이터 전처리 (Inception-v4 모델의 입력 크기 299x299 사용) ===
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'val': transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
}

# === 커스텀 데이터셋 클래스 (이미지와 JSON 파일에서 레이블 추출) ===
class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}

        # 폴더 이름을 이용해서 레이블 생성
        for class_name in os.listdir(image_dir):
            class_dir = os.path.join(image_dir, class_name)
            if os.path.isdir(class_dir):
                self.class_to_idx[class_name] = len(self.class_to_idx)
                for file in os.listdir(class_dir):
                    if file.endswith('.jpg'):
                        self.image_paths.append(os.path.join(class_dir, file))
                        self.labels.append(self.class_to_idx[class_name])

        print(f"Number of unique classes: {len(self.class_to_idx)}")
        print(f"Class to index mapping: {self.class_to_idx}")

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label)


# === 데이터셋 및 데이터로더 설정 ===
train_dataset = CustomImageDataset(image_dir=train_dir, transform=data_transforms['train'])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

val_dataset = CustomImageDataset(image_dir=val_dir, transform=data_transforms['val'])
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# === Inception-v4 모델 불러오기 (timm 라이브러리 사용) ===
model = create_model('inception_v4', pretrained=True, num_classes=num_classes)
model = model.to(device)
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
scaler = GradScaler()

# === Early Stopping 구현 ===
class EarlyStopping:
    def __init__(self, patience=10, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# === 체크포인트 로드 기능 ===
def load_checkpoint(model, optimizer, checkpoint_path):
    if os.path.isfile(checkpoint_path):
        print(f"=> loading checkpoint '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        best_acc = checkpoint['accuracy']
        print(f"=> loaded checkpoint '{checkpoint_path}' (epoch {epoch}, accuracy {best_acc})")
        return epoch, best_acc
    else:
        print(f"=> no checkpoint found at '{checkpoint_path}', starting from scratch")
        return 0, 0.0

# === 모델 학습 함수 ===
def train_model(model, criterion, optimizer, num_epochs, patience):
    start_epoch, best_acc = load_checkpoint(model, optimizer, checkpoint_path)
    early_stopping = EarlyStopping(patience=patience)
    
    print(f"\n학습 시작: 총 {num_epochs} 에포크")
    print(f"학습 데이터 크기: {len(train_dataset)} 샘플")
    print(f"검증 데이터 크기: {len(val_dataset)} 샘플")
    print(f"배치 크기: {batch_size}")
    print(f"Gradient Accumulation 스텝: {accumulation_steps}\n")
    
    for epoch in range(start_epoch, num_epochs):
        print(f"\n{'='*50}")
        print(f"에포크 {epoch+1}/{num_epochs} 시작")
        print(f"현재 학습률: {optimizer.param_groups[0]['lr']:.6f}")
        
        model.train()
        running_loss = 0.0
        running_corrects = 0

        optimizer.zero_grad()
        accumulation_counter = 0

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for inputs, labels in progress_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels) / accumulation_steps
            
            scaler.scale(loss).backward()
            accumulation_counter += 1

            if accumulation_counter % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0) * accumulation_steps
            running_corrects += torch.sum(preds == labels.data)
        
            progress_bar.set_postfix({
                'loss': f'{loss.item() * accumulation_steps:.4f}',
                'acc': f'{(torch.sum(preds == labels.data).double() / len(labels)):.4f}'
            })

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)

        print(f"\n[학습 결과]")
        print(f"평균 손실: {epoch_loss:.4f}")
        print(f"정확도: {epoch_acc:.4f} ({running_corrects}/{len(train_dataset)})")

        # 검증 단계
        print(f"\n[검증 단계 시작]")
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc='Validating'):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
        
        val_loss = val_loss / len(val_dataset)
        val_acc = val_corrects.double() / len(val_dataset)

        print(f"\n[검증 결과]")
        print(f"검증 손실: {val_loss:.4f}")
        print(f"검증 정확도: {val_acc:.4f} ({val_corrects}/{len(val_dataset)})")

        if val_acc > best_acc:
            best_acc = val_acc
            print(f"\n[새로운 최고 성능 모델 저장]")
            print(f"이전 최고 정확도: {best_acc:.4f}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': best_acc
            }, checkpoint_path)
        
        scheduler.step(val_loss)
        early_stopping(val_loss)
        
        if early_stopping.early_stop:
            print(f'\n[Early Stopping] 에포크 {epoch+1}에서 학습을 조기 종료합니다.')
            break
        
    return model

# === 모델 학습 ===
trained_model = train_model(model, criterion, optimizer, num_epochs, early_stopping_patience)

# 학습 완료 후 모델 저장
model_save_dir = './'
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)

model_save_path = os.path.join(model_save_dir, 'classification_cat_inception_v4_model.pth')
torch.save(trained_model.state_dict(), model_save_path)

print(f"Model saved to {model_save_path}")
