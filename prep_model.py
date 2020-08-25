import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import FeatureHasher
from sklearn.compose import make_column_transformer

dftrain = pd.read_csv('train.csv')
states = pd.read_csv("StateLabels.csv")
breeds = pd.read_csv("BreedLabels.csv")
colors = pd.read_csv("ColorLabels.csv")

# labelencoder = LabelEncoder()
# onehotencoder = OneHotEncoder()
# featureHasher = FeatureHasher(input_type="string")

dftrain.loc[dftrain['RescuerID'].value_counts()[dftrain['RescuerID']].values < 10, 'RescuerID'] = "Other Rescuers"

# dftrain['AgeCat'] = dftrain['Age'].apply(lambda x: 0 if x < 6 else
#                                              1 if x >= 6 and x < 12 else
#                                              2 if x >= 12 and x < 36 else
#                                              3 if x >= 36 and x < 60 else
#                                              4 if x >= 60 and x < 96 else
#                                              5);
# dftrain.pop("Age")

dftrain["HasName"] = 1
dftrain.loc[dftrain.Name.isnull(), "HasName"] = 0

# dftrain['HasVideos'] = dftrain['VideoAmt'].apply(lambda x: 1 if x > 0 else 0);
dftrain.pop('VideoAmt')

# dftrain['HasPhotos'] = dftrain['PhotoAmt'].apply(lambda x: 0 if x == 0 else (1 if x >= 1 and x <= 5 else 2));
# dftrain.pop('PhotoAmt')

dftrain['State'] = dftrain['State'].apply(lambda x: "State."+states.loc[states['StateID'] == x]['StateName'].iloc[0])
dftrain.loc[(dftrain["State"] != "Selangor") & (dftrain['State'] != 'Kuala Lumpur'),'State'] = "State.Other States"

mixed_breeds = [307, 264, 265, 266, 299]
dftrain['PureBreed'] = 'PureBreed.No'
dftrain.loc[((~dftrain['Breed1'].isin(mixed_breeds)) & (dftrain['Breed2'] == 0)) |
          ((~dftrain['Breed1'].isin(mixed_breeds)) & (dftrain['Breed1']==dftrain['Breed2'])) |
          ((dftrain['Breed1'] == 0) & (~dftrain['Breed2'].isin(mixed_breeds))),"PureBreed"] = 'PureBreed.Yes'
dftrain.loc[(dftrain['Breed1'] != 0) & 
          (~dftrain['Breed1'].isin(mixed_breeds)) & 
          (dftrain['Breed2'] != 0) & 
          (~dftrain['Breed2'].isin(mixed_breeds)),"PureBreed"] = 'PureBreed.Pure-Pure'
dftrain.loc[((~dftrain['Breed1'].isin(mixed_breeds)) & (dftrain['Breed2'].isin(mixed_breeds))) |
          ((dftrain['Breed2'] != 0) & (~dftrain['Breed2'].isin(mixed_breeds)) & (dftrain['Breed1'].isin(mixed_breeds))), "PureBreed"] = 'PureBreed.Pure-Not Pure'

dftrain['Breed1'] = dftrain['Breed1'].apply(lambda x: "Breed1."+breeds.loc[breeds['BreedID'] == x]['BreedName'].iloc[0])
dftrain.pop('Breed2')

dftrain.pop("PetID")
dftrain.pop('Name')
dftrain.pop('Health')
dftrain.pop('Description')

dftrain['Color1'] = dftrain['Color1'].apply(lambda x: "Color1." + colors.loc[colors['ColorID'] == x]['ColorName'].iloc[0]);
dftrain.pop('Color2')
dftrain.pop('Color3')

# if os.path.exists("train_model.csv"):
#   os.remove("train_model.csv")
# dftrain.to_csv('train_model.csv')

X = dftrain.drop('AdoptionSpeed', axis=1)
column_trans = make_column_transformer(
    (OneHotEncoder(), ['State','Color1','PureBreed','Breed1','RescuerID']),
    remainder='passthrough')
X = column_trans.fit_transform(X)

x_train, x_test, y_train, y_test = train_test_split(X, dftrain['AdoptionSpeed'])

scaler = StandardScaler(with_mean=False)
x_train_scaled = scaler.fit_transform(x_train) 
# x_train_scaled = pd.DataFrame(scaler.transform(x_train),columns = column_trans.get_feature_names())

scaler.fit(x_test)
x_test_scaled = scaler.fit_transform(x_test)
# x_test_scaled = pd.DataFrame(scaler.transform(x_test),columns = column_trans.get_feature_names())