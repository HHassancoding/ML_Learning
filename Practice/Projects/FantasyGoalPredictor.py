from sklearn.ensemble import HistGradientBoostingClassifier
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics  import accuracy_score
import numpy as np
from sklearn.model_selection import train_test_split


goalsP = pd.read_csv("../BegineerLearningPhase/fpl_player_statistics.csv", low_memory=False)

goalsP['Likely'] = goalsP.apply(lambda x:1 if(x['expected_goals_per_90']>0.4 or
                                             (x['goals_scored']>=6 and x['minutes']>=500)
                                             )else 0,
                                             axis =1
                                )

X = goalsP.drop(['Likely'], axis= 1)
y = goalsP['Likely']

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42,test_size=0.3)

model = HistGradientBoostingClassifier(
                                    max_depth=4,
                                    learning_rate = 0.1
                                      )
model.fit(X,y)

predictions = model.predict(X_test)

print("testing accuracy", accuracy_score(y_test, predictions))
