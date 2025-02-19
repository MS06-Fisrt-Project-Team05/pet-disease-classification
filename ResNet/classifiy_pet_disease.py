import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

data_dir = '../data/selected/dog'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
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

class_names = image_datasets['train'].classes
num_classes = len(class_names)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet50(weights=None)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        print('-' * 50)

        for phase in ['train', 'val']:
            if phase == 'train':
                print(f'\n훈련 단계 시작...')
                model.train()
            else:
                print(f'\n검증 단계 시작...')
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            
            # 진행률 표시를 위한 전체 배치 수 계산
            total_batches = len(dataloaders[phase])

            for batch_idx, (inputs, labels) in enumerate(dataloaders[phase]):
                # 현재 배치 진행률 표시
                if batch_idx % 10 == 0:  # 10배치마다 진행상황 출력
                    print(f'{phase} 진행률: {batch_idx}/{total_batches} 배치 처리 중... ({(100. * batch_idx / total_batches):.1f}%)')
                
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

            print(f'\n{phase.upper()} 결과:')
            print(f'손실값: {epoch_loss:.4f} 정확도: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                print(f'새로운 최고 정확도 달성! ({epoch_acc:.4f})')

        print()
    print('\n학습 완료!')
    print(f'최종 최고 검증 정확도: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    return model

num_epochs = 25
model = train_model(model, criterion, optimizer, scheduler, num_epochs=num_epochs)

torch.save(model.state_dict(), 'resnet50_dog_disease.pth')
print("Model saved to resnet50_dog_disease.pth")



