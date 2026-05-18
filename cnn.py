import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
torch.manual_seed(42)
train_data = torchvision.datasets.MNIST(
    root='data',
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

test_data = torchvision.datasets.MNIST(
    root='data',
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

model = nn.Sequential(
    nn.Conv2d(1, 32, 3),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(1600, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)
train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=64, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=64, shuffle=False
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 5
for epoch in range(epochs):
    for images, labels in train_loader:
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
        predictions = model(images)
        predicted_classes = torch.argmax(predictions, dim=1)
        correct += (predicted_classes == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total * 100
print(f"Accuracy: {accuracy:.2f}%")

# then single image prediction
image, label = test_data[96]
with torch.no_grad():
    prediction = model(image.unsqueeze(0))
    predicted = torch.argmax(prediction).item()

plt.imshow(image.squeeze(), cmap='gray')
plt.title(f"Actual: {label}  Predicted: {predicted}")
plt.show()
with torch.no_grad():
    for i in range(1000):
        image, label = test_data[i]
        prediction = model(image.unsqueeze(0))
        predicted = torch.argmax(prediction).item()
        if predicted != label:
            plt.imshow(image.squeeze(), cmap='gray')
            plt.title(f"Actual: {label}  Predicted: {predicted}")
            plt.show()
            break