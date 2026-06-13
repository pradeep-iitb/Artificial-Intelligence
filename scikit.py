'''
Supervised Learning is a type of machine learning where the model is trained on labeled data. In supervised learning, the algorithm learns from the input-output pairs and makes predictions based on that learning. The goal is to learn a mapping from inputs to outputs.

Unsupervised Learning, on the other hand, is a type of machine learning where the model is trained on unlabeled data. In unsupervised learning, the algorithm tries to find patterns or structure in the data without any predefined labels. The goal is to discover hidden patterns or groupings in the data.

Reinforcement Learning is a type of machine learning where an agent learns to make decisions by interacting with an environment. The agent receives feedback in the form of rewards or penalties based on its actions, and it learns to maximize the cumulative reward over time. The goal is to learn a policy that maps states to actions in order to achieve the best long-term outcome.
'''
from sklearn.linear_model import LinearRegression

X = [[1], [2], [3], [4], [5]]
y = [40, 50, 65, 75, 90]

model = LinearRegression()
model.fit(X, y)
hour = float(input("Enter hours studied: "))
predicted_marks = model.predict([[hour]])
print(f"Predicted marks for {hour} hours of study: {predicted_marks}")