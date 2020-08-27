import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree, model_selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import math

from prep_model import dftrain,x_train_scaled,x_test_scaled,x_train,x_test,y_train,y_test,column_trans,one_hot_cols

generalized_tree = tree.DecisionTreeClassifier(
    random_state = 1,
    max_depth=10,
    min_samples_split = 2
)
generalized_tree.fit(x_train_scaled, y_train)

print(generalized_tree.score(x_test_scaled, y_test))

importances = generalized_tree.feature_importances_
indices = np.argsort(importances)[::-1]

numOfFeatures = 30

indices = indices[0:numOfFeatures]

feature_names = [column_trans.get_feature_names()[i] for i in indices]
for i,v in enumerate(one_hot_cols):
    matchingStr = 'onehotencoder__x'+str(i)+'_'
    for j,feature in enumerate(feature_names):
        if matchingStr in feature:
            feature_names[j] = feature.replace(matchingStr, v+'.')

fig = plt.figure(figsize=(14, 6))
plt.title("Decision Tree Feature Importance")
plt.barh(range(numOfFeatures), importances[indices], align='center')
plt.yticks(range(numOfFeatures), feature_names)
plt.tight_layout()
plt.show()