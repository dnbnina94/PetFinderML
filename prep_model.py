import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import FeatureHasher
from sklearn.compose import make_column_transformer

dftrain = pd.read_csv('train_prep.csv')
dftest = pd.read_csv('test_prep.csv')
dftrain = dftrain.drop(dftrain.columns[0], axis=1)
dftest = dftest.drop(dftest.columns[0], axis=1)

dropCols = ['PetID','Name','Description','BadName','Color2','Color3', 
              'VideoAmt','Colorful','HasDesc','FeeCat','AgeCat','HasPhotos','Breed2','HasVideos','State']

dftrain = dftrain.drop(dropCols, axis=1)
dftest = dftest.drop(dropCols, axis=1)

one_hot_cols = ['StateCat','Color1','Breed1','PureBreed','RescuerID','Health','Gender','Type',
                'MaturitySize','FurLength','Vaccinated','Dewormed','Sterilized','HasName','Language']

X = dftrain.drop('AdoptionSpeed', axis=1)
column_trans = make_column_transformer(
    (OneHotEncoder(handle_unknown='ignore'), one_hot_cols),
    remainder='passthrough')
X = column_trans.fit_transform(X)

X_test = column_trans.transform(dftest)

x_train, x_test, y_train, y_test = train_test_split(X, dftrain['AdoptionSpeed'])

scaler = StandardScaler(with_mean=False)
x_train_scaled = scaler.fit_transform(x_train) 
# x_train_scaled = pd.DataFrame(scaler.transform(x_train),columns = column_trans.get_feature_names())

scaler.fit(x_test)
x_test_scaled = scaler.fit_transform(x_test)
# x_test_scaled = pd.DataFrame(scaler.transform(x_test),columns = column_trans.get_feature_names())

x_real_test_scaled = scaler.fit_transform(X_test) 