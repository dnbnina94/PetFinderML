import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn import tree, model_selection
import math

from prep_model import dftrain,x_train_scaled,x_test_scaled,x_train,x_test,y_train,y_test,column_trans,one_hot_cols,x_real_test_scaled

generalized_tree = tree.DecisionTreeClassifier(
    # random_state = 1,
    max_depth=5,
    # min_samples_split = 2
)
generalized_tree.fit(x_train_scaled, y_train)

result = generalized_tree.predict(x_real_test_scaled)
sample_submission = pd.read_csv('sample_submission.csv')
sample_submission['AdoptionSpeed'] = result

if os.path.exists('./submissions/dec_tree_submission.csv'):
  os.remove('./submissions/dec_tree_submission.csv')
sample_submission.to_csv('./submissions/dec_tree_submission.csv', index=False)

print("Decision Tree Score: ", generalized_tree.score(x_test_scaled, y_test))

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