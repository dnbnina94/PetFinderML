import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from prep_model import dftrain,x_train_scaled,x_test_scaled,x_train,x_test,y_train,y_test,column_trans

linReg = LinearRegression()
linReg.fit(x_train_scaled,y_train)

print (linReg.score(x_test_scaled, y_test))
