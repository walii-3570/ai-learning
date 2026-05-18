import torch
import torch.nn as nn

inputs =torch.tensor([
    [2.0,60.0],
    [4.0,70.0],
    [6.0,80.0],
    [8.0,90.0],
    [10.0,95.0]])
outputs=torch.tensor([[0.0],[0.0],[1.0],[1.0],[1.0]])
inputs =inputs/inputs.max(dim=0).values

model=nn.Sequential(
    nn.Linear(2,3),
    nn.Sigmoid(),
    nn.Linear(3,1),
    nn.Sigmoid()
)
criterion=nn.BCELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.1)

epochs=10000
for i in range(epochs):
    prediction=model(inputs)
    loss=criterion(prediction,outputs)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

with torch.no_grad():
    predictions = model(inputs)
    print("\nResults:")
    for i in range(len(inputs)):
        pred = predictions[i][0].item()
        result = "PASS" if pred > 0.5 else "FAIL"
        print(f"Student {i+1}: {pred:.2f} → {result}")