import numpy as np
inputs =np.array([
    [2, 60],
    [4, 70],
    [6, 80],
    [8, 90],
    [10, 95]
])
outputs = np.array([0, 0, 1, 1, 1])
# start with random weights and bias
w = np.array([0.0, 0.0])
b = 0.0

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def predict(input_row):
    result = np.sum(w*input_row)+b
    return sigmoid(result)
print(predict(inputs[0]))
learning_rate=0.01
epochs=10000
for i in range(epochs):
    for j in range(len(inputs)):
        predicted=predict(inputs[j])
        error=predicted-outputs[j]
        w[0]= w[0]- (learning_rate*error*inputs[j][0])
        w[1]= w[1]- (learning_rate*error*inputs[j][1])
        b= b- (learning_rate*error)
        print(f"Final weights: {w}")
print(f"Final bias: {b}")
print("\nPredictions:")
for i in range(len(inputs)):
    pred = predict(inputs[i])
    result = "PASS" if pred > 0.5 else "FAIL"
    print(f"Student {i+1}: {pred:.2f} → {result}")
new_student = np.array([5, 75])  # 5 hours, 75% attendance
pred = predict(new_student)
result = "PASS" if pred > 0.5 else "FAIL"
print(f"New student prediction: {pred:.2f} → {result}")