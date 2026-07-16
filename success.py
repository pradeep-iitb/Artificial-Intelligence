# Steps for the project 
'''
1. Loading the Data - Gather and Analyse the dataset to understand its structure and content.
2. Data Preprocessing - Handle missing values, outliers and inconsistencies to ensure data quality.
3. Feature Scaling - Normalize or standardize features to improve model performance 
4. Split the data - Divide the dataset into training , validation and test sets
5. Train a model - Select and train a machine learning model on the training data
6. Make Predictions - Use the trained model to make predictions on new data
7. Evaluate the model - Assess the performance of the model using appropriate metrics
8. Visualize Results - Create visualizations to communicate the findings and insights from the model
9. Improve the Model - Based on the evaluation results, make necessary adjustments to improve the model's performance
10. Wrap up - Summarize the findings, document the process and share the results with stakeholders.
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder , StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('student-success-data.csv')

# print("Sample Rows:")
# print(df.head())

# print("DataSet Shape:")
# print(f'Rows: {df.shape[0]}, Columns: {df.shape[1]}')

# print("Summary Statistics:")
# print(df.describe())

# print("Missing Values:")
# missing_values = df.isnull().sum()

print ("Missing Values in each column:")
print(df.isnull().sum())

le = LabelEncoder()

df['InternetAccess'] = le.fit_transform(df['InternetAccess'])
df['Passed'] = le.fit_transform(df['Passed'])

# print("Encoded Data:")
# print(df.head())

# Feature Scaling

features = ['StudyHours', 'Attendance','PastScore','SleepHours', 'InternetAccess']

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[features])

X = df_scaled[features]
y = df_scaled['Passed']
X_train , X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print('Classification Report:')
print(classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix , annot=True, fmt='d', cmap='Blues',xticklabels=['Fail' , 'Pass'], yticklabels=['Fail', 'Pass'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

print("------Predict Your Results------")

try:
    study_hours = float(input("Enter Study Hours: "))
    attendance = float(input("Enter Attendance Percentage: "))
    past_score = float(input("Enter Past Score: "))
    sleep_hours = float(input("Enter Sleep Hours: "))
    internet_access = input("Do you have Internet Access? (yes/no): ").strip().lower()
    
    user_input_df = pd.DataFrame([{
        'StudyHours': study_hours,
        'Attendance': attendance,
        'PastScore': past_score,
        'SleepHours': sleep_hours,
        'InternetAccess': 1 if internet_access == 'yes' else 0  # Encoding internet access     
    }])

    user_input_scaled = scaler.transform(user_input_df[features])
    user_prediction = model.predict(user_input_scaled)

    result = 'Pass' if user_prediction[0] == 1 else 'Fail'
    print(f"Prediction based on your input: {result}")
except Exception as e:
    print(f"Error in input: {e}")

