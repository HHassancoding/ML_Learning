# Improved Streaming Project - Random Forest on Kaggle Dataset

import pandas as pd
import ast
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# ---------------------------
# 1️⃣ Load datasets
# ---------------------------
movies = pd.read_csv("movies_metadata.csv", low_memory=False)
ratings = pd.read_csv("ratings.csv")

# ---------------------------
# 2️⃣ Fix IDs and numeric conversion
# ---------------------------
movies.rename(columns={'id':'movieId'}, inplace=True)
movies['movieId'] = pd.to_numeric(movies['movieId'], errors='coerce')
ratings['movieId'] = pd.to_numeric(ratings['movieId'], errors='coerce')

# Drop rows with invalid movieId
movies = movies.dropna(subset=['movieId'])
ratings = ratings.dropna(subset=['movieId'])

# ---------------------------
# 3️⃣ Merge movies and ratings
# ---------------------------
df = pd.merge(ratings, movies, on='movieId')

# Optional: sample 100k rows to reduce memory usage
df = df.sample(n=70000, random_state=42)

# ---------------------------
# 4️⃣ Create target column: Liked
# ---------------------------
df['Liked'] = df['rating'].apply(lambda x: 1 if x >= 4 else 0)

# ---------------------------
# 5️⃣ Process genres into numeric columns
# ---------------------------
def extract_main_genres(genres_str):
    try:
        genres_list = ast.literal_eval(genres_str)
        return [g['name'] for g in genres_list]
    except:
        return []

df['main_genres'] = df['genres'].apply(extract_main_genres)

top_genres = ['Action','Comedy','Drama','Romance','Horror','Thriller',
              'Adventure','Animation','Crime','Fantasy']

for genre in top_genres:
    df[genre] = df['main_genres'].apply(lambda x: 1 if genre in x else 0)

# ---------------------------
# 6️⃣ Add more numeric features
# ---------------------------
# Convert 'popularity' and 'vote_count' to numeric
df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce')
df['vote_count'] = pd.to_numeric(df['vote_count'], errors='coerce')

# Fill NaN values with 0
df['popularity'] = df['popularity'].fillna(0)
df['vote_count'] = df['vote_count'].fillna(0)

# ---------------------------
# 7️⃣ Define features (X) and target (y)
# ---------------------------
X = df[top_genres + ['popularity', 'vote_count']]   # now 12 features
y = df['Liked']

# ---------------------------
# 8️⃣ Split into train/test sets
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ---------------------------
# 9️⃣ Train Random Forest Classifier
# ---------------------------
rf_model = RandomForestClassifier(
    n_estimators=300,   # more trees
    max_depth=None,     # let trees grow deeper
    random_state=42
)

rf_model.fit(X_train, y_train)

# ---------------------------
# 🔟 Make predictions & check accuracy
# ---------------------------
y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)

print("Training Accuracy:", accuracy_score(y_train, y_train_pred))
print("Testing Accuracy:", accuracy_score(y_test, y_test_pred))

# ---------------------------
# 1️⃣1️⃣ Feature importance
# ---------------------------
plt.figure(figsize=(10,6))
plt.barh(X.columns, rf_model.feature_importances_)
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importance for Predicting Liked Movies")
plt.show()
