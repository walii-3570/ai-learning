import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import models
model=models.resnet18(weights='IMAGENET1K_V1')

model.fc=nn.Linear(512,2)
# freeze all layers except last one
for param in model.parameters():
    param.requires_grad = False

# only train the last layer we added
model.fc.requires_grad_(True)
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])
train_data = torchvision.datasets.CIFAR10(
    root='data', train=True, download=True, transform=transform
)
test_data = torchvision.datasets.CIFAR10(
    root='data', train=False, download=True, transform=transform
)
def filter_classes(dataset, classes):
    indices = [i for i, (_, label) in enumerate(dataset) 
               if label in classes]
    return torch.utils.data.Subset(dataset, indices)

train_data = filter_classes(train_data, [3, 5])
test_data = filter_classes(test_data, [3, 5])

print(f"Training samples: {len(train_data)}")
print(f"Testing samples: {len(test_data)}")
train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=32, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=32, shuffle=False
)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
epochs=3
for epoch in range(epochs):
    for images, labels in train_loader:
        labels = (labels == 5).long()
        predictions = model(images)
        loss = criterion(predictions, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} done")
    correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        labels = (labels == 5).long()
        predictions = model(images)
        predicted_classes = torch.argmax(predictions, dim=1)
        correct += (predicted_classes == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total * 100
print(f"Accuracy: {accuracy:.2f}%")