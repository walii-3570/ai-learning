import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
hours =[2,4,6,8,10]
scores =[40,55,65,75,90]
plt.scatter(hours,scores)
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.title("Hours vs Scores")
m,b=np.polyfit(hours,scores,1)
plt.plot(hours,[m*x +b for x in hours],color='black')
plt.show()
hours_studied =7
predicted_score=m*hours_studied +b
print(f"predicted score for 7 hours: {predicted_score}")
print(f"Slope (m): {m}")
print(f"Intercept (b): {b}")