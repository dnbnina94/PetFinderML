import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# from sklearn import preprocessing

dftrain = pd.read_csv('train.csv')

dftrain.loc[dftrain['RescuerID'].value_counts()[dftrain['RescuerID']].values < 5, 'RescuerID'] = "Other"

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

dftrain['State'] = dftrain['State'].apply(lambda x: 1 if x == 41326 else
                                                   (2 if x == 41401 else 3))

mixed_breeds = [307, 264, 265, 266, 299]
dftrain['PureBreed'] = 0
dftrain.loc[((~dftrain['Breed1'].isin(mixed_breeds)) & (dftrain['Breed2'] == 0)) |
          ((~dftrain['Breed1'].isin(mixed_breeds)) & (dftrain['Breed1']==dftrain['Breed2'])) |
          ((dftrain['Breed1'] == 0) & (~dftrain['Breed2'].isin(mixed_breeds))),"PureBreed"] = 1
dftrain.loc[(dftrain['Breed1'] != 0) & 
          (~dftrain['Breed1'].isin(mixed_breeds)) & 
          (dftrain['Breed2'] != 0) & 
          (~dftrain['Breed2'].isin(mixed_breeds)),"PureBreed"] = 2
dftrain.loc[((~dftrain['Breed1'].isin(mixed_breeds)) & (dftrain['Breed2'].isin(mixed_breeds))) |
          ((dftrain['Breed2'] != 0) & (~dftrain['Breed2'].isin(mixed_breeds)) & (dftrain['Breed1'].isin(mixed_breeds))), "PureBreed"] = 3
dftrain.pop('Breed1')
dftrain.pop('Breed2')

# dftrain.loc[dftrain.Quantity > 6, "Quantity"] = 6
dftrain.pop("PetID")
dftrain.pop('Name')
dftrain.pop('Health')
dftrain.pop('Description')

# dftrain.pop('Color1')
dftrain.pop('Color2')
dftrain.pop('Color3')

# NOT SURE
# dftrain.pop("RescuerID")

one_hot = pd.get_dummies(dftrain['RescuerID'])
dftrain = dftrain.drop('RescuerID',axis = 1)
dftrain = dftrain.join(one_hot)

x_train, x_test, y_train, y_test = train_test_split(dftrain.drop('AdoptionSpeed', axis=1), dftrain['AdoptionSpeed'])

scaler = StandardScaler()
scaler.fit(x_train) 
x_train_scaled = pd.DataFrame(scaler.transform(x_train),columns = x_train.columns)

logReg = LogisticRegression(solver="sag")
logReg.fit(x_train_scaled, y_train)

scaler.fit(x_test)
x_test_scaled = pd.DataFrame(scaler.transform(x_test),columns = x_test.columns)

print(logReg.score(x_test_scaled, y_test))

# feature_importance = abs(logReg.coef_[0])
# feature_importance = 100.0 * (feature_importance / feature_importance.max())
# sorted_idx = np.argsort(feature_importance)
# pos = np.arange(sorted_idx.shape[0]) + .5

# fig = plt.figure(figsize=(14, 6))
# featax = fig.add_subplot(1, 1, 1)
# featax.barh(pos, feature_importance[sorted_idx], align='center')
# featax.set_yticks(pos)
# featax.set_yticklabels(np.array(x_train.columns)[sorted_idx], fontsize=8)
# featax.set_xlabel('Relative Feature Importance')
# plt.tight_layout()   
# plt.show()
