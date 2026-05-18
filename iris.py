import torch
import torch.nn as nn
torch.manual_seed(42)
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris=load_iris()
x=torch.tensor(iris.data, dtype=torch.float32)
y=torch.tensor(iris.target,dtype=torch.long)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
print(f"Training samples: {len(x_train)}")
print(f"Testing samples: {len(x_test)}")
model =nn.Sequential(
    nn.Linear(4,8),
    nn.ReLU(),
    nn.Linear(8,3),

)
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.01)
epochs=100
for i in range(epochs):
    model.train()
    prediction = model(x_train)
    loss=criterion(prediction,y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
with torch.no_grad():
    predictions=model(x_test)
    predicted_classes=torch.argmax(predictions,dim=1)
    correct=(predicted_classes==y_test).sum().item()
    accuracy=correct/len(y_test)*100
    print(f"Accuracy: {accuracy}%")

