import numpy as np
inputs =np.array([
      [2, 60],
    [4, 70],
    [6, 80],
    [8, 90],
    [10, 95]
])
inputs = inputs / inputs.max(axis=0)
outputs =np.array([[0], [0], [1], [1], [1]])
def sigmoid(x):
    return 1/(1+np.exp(-x))
w1=np.random.randn(2,3)
w2=np.random.randn(3,1)
b1=np.zeros((1,3))
b2=np.zeros((1,1))
def forward(inputs):
    hidden = sigmoid(np.dot(inputs, w1) + b1)
    output = sigmoid(np.dot(hidden, w2) + b2)
    return output

learning_rate = 0.1
epochs = 10000
for i in range(epochs):
    hidden = sigmoid(np.dot(inputs, w1) + b1)
    output = sigmoid(np.dot(hidden, w2) + b2)
    error =outputs-output
    d_output=error*output*(1-output)
    error_hidden=d_output.dot(w2.T)
    d_hidden=error_hidden*hidden*(1-hidden)
    w2 += hidden.T.dot(d_output)*learning_rate
    w1 += inputs.T.dot(d_hidden)*learning_rate

new_student = np.array([[8, 10]])
new_student = new_student / np.array([10, 95])  # same max values as before
pred = forward(new_student)[0][0]
result = "PASS" if pred > 0.5 else "FAIL"
print(f"New student (8hrs, 10%): {pred:.2f} → {result}")