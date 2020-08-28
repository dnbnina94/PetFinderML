import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from prep_model import dftrain,x_train_scaled,x_test_scaled,x_train,x_test,y_train,y_test,column_trans,one_hot_cols

def show_plot(index):
    indicies = np.argsort(abs(svm.coef_[index]))[0:30]
    feature_names = [column_trans.get_feature_names()[i] for i in indicies]

    for i,v in enumerate(one_hot_cols):
        matchingStr = 'onehotencoder__x'+str(i)+'_'
        for j,feature in enumerate(feature_names):
            if matchingStr in feature:
                feature_names[j] = feature.replace(matchingStr, v+'.')

    coefficients = [svm.coef_[index][i] for i in indicies]
    print(coefficients)

    fig = plt.figure(figsize=(14, 6))
    plt.title("Support Vector Machine Class " + str(index))
    plt.bar(feature_names, coefficients)
    plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right');
    plt.tight_layout()

svm = LinearSVC(max_iter=5000)
# svm = SVC(kernel='linear',gamma='auto')
svm.fit(x_train_scaled, y_train)

print("Support Vector Machine score:", svm.score(x_test_scaled, y_test))

for i in [0,1,2,3,4]:
    show_plot(i)

plt.show()