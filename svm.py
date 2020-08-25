import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from prep_model import dftrain,x_train_scaled,x_test_scaled,x_train,x_test,y_train,y_test,column_trans

svc = SVC(gamma='auto')
svc.fit(x_train_scaled, y_train)

print("Support Vector Machine score:", svc.score(x_test_scaled, y_test))