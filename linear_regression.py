import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

dftrain = pd.read_csv('train.csv')

dftrain['AgeCat'] = dftrain['Age'].apply(lambda x: 0 if x < 6 else
                                             1 if x >= 6 and x < 12 else
                                             2 if x >= 12 and x < 36 else
                                             3 if x >= 36 and x < 60 else
                                             4 if x >= 60 and x < 96 else
                                             5);
dftrain["HasName"] = 1
dftrain.loc[dftrain.Name.isnull(), "HasName"] = 0
dftrain['HasVideos'] = dftrain['VideoAmt'].apply(lambda x: 1 if x > 0 else 0);
dftrain['HasPhotos'] = dftrain['PhotoAmt'].apply(lambda x: 1 if x > 0 else 0);

dftrain.pop("Age")
dftrain.pop("PetID")
dftrain.pop('VideoAmt')
dftrain.pop('PhotoAmt')
dftrain.pop('Name')
dftrain.pop('Health')
dftrain.pop('Description')

# NOT SURE
dftrain.pop("RescuerID")

x_train, x_test, y_train, y_test = train_test_split(dftrain.drop('AdoptionSpeed', axis=1), dftrain['AdoptionSpeed'])

linReg = LinearRegression()
linReg.fit(x_train,y_train)

print (linReg.score(x_test, y_test))
