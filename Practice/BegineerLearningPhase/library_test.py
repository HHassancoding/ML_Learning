from pyexpat import features
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt


# Load iris dataset
iris = load_iris()
x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=42)


# Put it into a DataFrame
df = pd.DataFrame(x_train, columns=iris.feature_names)
df['target'] = y_train

print(df.head())

# Create a model
model = RandomForestClassifier(n_estimators=100, random_state=42)


# Train the model
model.fit(x_train, y_train)


importance = model.feature_importances_
indices = np.argsort(importance)[::-1]


predictions = model.predict(x_test)
train_predictions = model.predict(x_train)
accuracy = accuracy_score(y_test, predictions)
train_accuracy = accuracy_score(y_train, train_predictions)


# Test predictions
print("Predictions:", predictions)
print("Actual:", y_test)
print("accuracy :", accuracy )
print("training accuracy: " , train_accuracy)

plt.figure(figsize=(8,6))
plt.title("Feature Importances")
plt.bar(range(x.shape[1]), importance[indices], color="skyblue", align="center")
plt.xticks(range(x.shape[1]), [iris.feature_names[i] for i in indices], rotation=45)
plt.show()

