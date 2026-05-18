hours =  [2, 4, 6, 8, 10]
scores = [40, 55, 65, 75, 90]

m = 0
b = 0
learning_rate = 0.001
epochs=100000
for i in range(epochs):
    for x,y in zip(hours,scores):
        predicted=m*x+b
        error=predicted-y
        m=m-(learning_rate *error*x)
        b=b-(learning_rate *error)

print(f"Final m: {m}")
print(f"Final b: {b}")
import matplotlib.pyplot as plt

plt.scatter(hours, scores, color='blue', label='Actual')
plt.plot(hours, [m*x + b for x in hours], color='red', label='Our line')
plt.xlabel("Hours Studied")
plt.ylabel("Score")
plt.title("Gradient Descent Result")
plt.legend()
plt.show()