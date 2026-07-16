from sklearn.linear_model import LinearRegression

X = [[1], [2], [3], [4], [5]]
y = [40, 50, 65, 75, 90] 

model = LinearRegression()
model.fit(X, y)
hour = float(input("Enter hours studied: "))
predicted_marks = model.predict([[hour]])
print(f"Predicted marks for {hour} hours of study: {predicted_marks}")

from sklearn.linear_model import LogisticRegression

X = [[1], [2], [3], [4], [5]]
y = [0, 0, 1, 1, 1]

model = LogisticRegression()
model.fit(X, y)
hour = float(input("Enter hours studied: "))
predicted_pass = model.predict([[hour]])
print(f"Predicted pass/fail for {hour} hours of study: {'Pass' if predicted_pass[0] == 1 else 'Fail'}")

from sklearn.neighbors import KNeighborsClassifier

X = [[1], [2], [3], [4], [5]]
y = [0, 0, 1, 1, 1]

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)
hour = float(input("Enter hours studied: "))
predicted_pass = model.predict([[hour]])
print(f"Predicted pass/fail for {hour} hours of study: {'Pass' if predicted_pass[0] == 1 else 'Fail'}")

from sklearn.linear_model import KNeighborsClassifier

X = [[180, 7] , [200,7.5], [250, 8],[300, 8.5],[330, 9],[360, 9.5]]

y = [0, 0, 1, 1, 1, 1]

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)
weight = float(input("Enter weight (in gms): "))
size = float(input("Enter size (in cm): "))
predicted_fruit = model.predict([[weight, size]])
print(f"Predicted fruit for weight {weight} gms and size {size} cm: {'Apple' if predicted_fruit[0] == 0 else 'Orange'}")

from sklearn.metrics import confusion_matrix

y_true = [1,0,1,1,0,1,0,0,1,0]
y_pred = [1,0,1,0,0,0,1,0,1,0]

cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

'''
True Negative (TN): 4    False Positive (FP): 1
False Negative (FN): 2   True Positive (TP): 3
'''

'''
NAE - Normalized Absolute Error
MSE - Mean Squared Error
RMSE - Root Mean Squared Error
'''

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





