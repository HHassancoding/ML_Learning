import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

# Load data
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Features and target
X = df.drop('target', axis=1)
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = DecisionTreeClassifier(max_depth=6,random_state=42)
model.fit(X_train, y_train)

# Predictions
trainingPredict = model.predict(X_train)
testPredict = model.predict(X_test)

# Accuracy
print("Training accuracy:", accuracy_score(y_train, trainingPredict))
print("Testing accuracy:", accuracy_score(y_test, testPredict))

# Feature importance
importance = model.feature_importances_
features = X_train.columns
