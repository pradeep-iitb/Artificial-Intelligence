import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error , mean_squared_error,r2_score
import matplotlib.pyplot as plt
import numpy as np

data = pd. read_csv("Student-Data.csv")

X = data[['Studys_hours_per_week']]
y = data['Final_Score']

model = LinearRegression()
model.fit(X,y)

predicted_score = model.predict(X)

#Valid Regression Matrix

mae = mean_absolute_error(y, predicted_score)
mse = mean_squared_error(y, predicted_score)
rmse = np.sqrt(mse)
r2 = r2_score(y,predicted_score)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R^2 Score (Model Accuracy)' : {r2} ") # closer to 1 means better model accuracy and it better describes the variation

plt.figure(figsize=(10,6))
plt.hist(data['Final_Score'], bins=50 , color='skyblue', edgecolor='black')
plt.title('Distribution of Final Scores')
plt.xlabel('Final Exam Score')
plt.ylabel('Number of Students')
plt.grid(True)
plt.show()


plt.figure(figsize=(10,6))
plt.scatter(X,y, color='skyblue', label='Final Scores')
plt.plot(X, predicted_score, color='red', label='Predicted Scores(Regression Line)')
plt.title('Model Prediction vs Actual Scores')
plt.xlabel('Study Hours per Week')
plt.ylabel('Final Output Score')
plt.grid(True)
plt.show()

new_hour = 9 
predicted_new_score = model.predict([[new_hour]])
print(f"Predicted score for {new_hour} hours of study: {predicted_new_score[0]}")