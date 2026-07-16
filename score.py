import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression

data = pd.read_csv('student.csv')

X = data[['Hours']] # why use double brackets here - The double brackets `[['Hours']]` are used to select a column from a pandas DataFrame and return it as a DataFrame rather than a Series.  
y = data['Score']

model = LinearRegression()
model.fit(X, y)

predicted_score = model.predict(X)

mae = mean_absolute_error(y, predicted_score)
mse = mean_squared_error(y, predicted_score)
rmse = np.sqrt(mse)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

new_hour = float(input("Enter hours studied for new prediction: "))
new_pred = model.predict([[new_hour]])
print(f"Predicted score for {new_hour} hours of study: {new_pred[0]}")

'''
3-Step Foormula
1- X and y numeric ? Scatter plot - relationships
2- Is one Column  a category ? - Box plot or Count plot
3- Want to see distribution shape ? Histogram / kde plot / box plot
'''
