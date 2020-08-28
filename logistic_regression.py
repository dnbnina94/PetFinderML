import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from prep_model import dftrain,x_train_scaled,x_test_scaled,x_train,x_test,y_train,y_test,column_trans,one_hot_cols,x_real_test_scaled

def show_plot(index):
    indicies = np.argsort(abs(logReg.coef_[index]))[0:30]
    feature_names = [column_trans.get_feature_names()[i] for i in indicies]

    for i,v in enumerate(one_hot_cols):
        matchingStr = 'onehotencoder__x'+str(i)+'_'
        for j,feature in enumerate(feature_names):
            if matchingStr in feature:
                feature_names[j] = feature.replace(matchingStr, v+'.')

    coefficients = [logReg.coef_[index][i] for i in indicies]

    fig = plt.figure(figsize=(14, 6))
    plt.title("Logistic Regression Class " + str(index))
    plt.bar(feature_names, coefficients)
    plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right');
    plt.tight_layout()


logReg = LogisticRegression()
logReg.fit(x_train_scaled, y_train)

print(len(logReg.predict(x_real_test_scaled)))

print("Logistic Regression score:", logReg.score(x_test_scaled, y_test))

for i in [0,1,2,3,4]:
    show_plot(i)

plt.show()

