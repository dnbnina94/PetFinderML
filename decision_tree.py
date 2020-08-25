import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree, model_selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import math

from prep_model import dftrain,x_train_scaled,x_test_scaled,x_train,x_test,y_train,y_test,column_trans

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

names = [column_trans.get_feature_names()[i] for i in indices]

fig = plt.figure(figsize=(14, 6))
plt.title("Decision Tree Feature Importance")
plt.barh(range(numOfFeatures), importances[indices], align='center')
plt.yticks(range(numOfFeatures), names)
plt.tight_layout()
plt.show()