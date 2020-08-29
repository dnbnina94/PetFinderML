import pandas as pd
import numpy as np
import math
import os

print('Enter file name:')
x = input()

df = pd.read_csv(x+".csv")
imgFeatures = pd.read_csv(x+'_img_features.csv')
imgFeatures = imgFeatures.drop(['PetID'], axis=1)
dfSentiment = pd.read_csv(x+'_sentiment_features.csv')
dfSentiment = dfSentiment.drop(['PetID'], axis=1)
states = pd.read_csv("StateLabels.csv")
breeds = pd.read_csv("BreedLabels.csv")
colors = pd.read_csv("ColorLabels.csv")

df['Type'] = df['Type'].apply(lambda x: "dogs" if x == 1 else "cats")
df['Color1'] = df['Color1'].apply(lambda x: colors.loc[colors['ColorID'] == x]['ColorName'].iloc[0])
df['Color2'] = df['Color2'].apply(lambda x: colors.loc[colors['ColorID'] == x]['ColorName'].iloc[0])
df['Color3'] = df['Color3'].apply(lambda x: colors.loc[colors['ColorID'] == x]['ColorName'].iloc[0])
df['State'] = df['State'].apply(lambda x: states.loc[states['StateID'] == x]['StateName'].iloc[0])
df['StateCat'] = df['State'].apply(lambda x: 'Other' if (x != 'Selangor' and x != 'Kuala Lumpur') else x)
df['Breed1'] = df['Breed1'].apply(lambda x: breeds.loc[breeds['BreedID'] == x]['BreedName'].iloc[0])
df['Breed1'].fillna('')
df['Breed2'] = df['Breed2'].apply(lambda x: breeds.loc[breeds['BreedID'] == x]['BreedName'].iloc[0])
df['Breed2'].fillna('')

mixed_breeds = ['Mixed Breed', 'Domestic Short Hair', 'Domestic Medium Hair', 'Domestic Long Hair', 'Tabby']
df['PureBreed'] = 'No'
df.loc[((~df['Breed1'].isin(mixed_breeds)) & (df['Breed2'].isnull())) |
          ((~df['Breed1'].isin(mixed_breeds)) & (df['Breed1']==df['Breed2'])) |
          ((df['Breed1'].isnull()) & (~df['Breed2'].isin(mixed_breeds))),"PureBreed"] = 'Yes'
df.loc[(df['Breed1'].notnull()) & 
          (~df['Breed1'].isin(mixed_breeds)) & 
          (df['Breed2'].notnull()) & 
          (~df['Breed2'].isin(mixed_breeds)),"PureBreed"] = 'Pure-Pure';
df.loc[((~df['Breed1'].isin(mixed_breeds)) & (df['Breed2'].isin(mixed_breeds))) |
          ((df['Breed2'].notnull()) & (~df['Breed2'].isin(mixed_breeds)) & (df['Breed1'].isin(mixed_breeds))), "PureBreed"] = 'Pure-Not Pure'

df['Colorful'] = 'One Color'
df.loc[(df['Color2'] != "Unknown") & (df['Color3'] != "Unknown"), 'Colorful'] = 'Three Colors'
df.loc[(df['Color2'] != "Unknown") & (df['Color3'] == "Unknown"), 'Colorful'] = 'Two Colors'

df["HasName"] = 'Yes'
df.loc[df.Name.isnull(), "HasName"] = 'No'
df['BadName'] = df['Name'].apply(lambda x: 'Yes' if len(str(x)) <= 2 else 'No')
df['HasVideos'] = df['VideoAmt'].apply(lambda x: 'Yes' if x > 0 else 'No')
df['HasPhotos'] = df['PhotoAmt'].apply(lambda x: 0 if x == 0 else 1)
df['Gender'] = df['Gender'].apply(lambda x: "male" if x == 1 else ("female" if x == 2 else "mixed"))
df['Vaccinated'] = df['Vaccinated'].apply(lambda x: "yes" if x == 1 else ("no" if x == 2 else "not sure"))
df['Dewormed'] = df['Dewormed'].apply(lambda x: "yes" if x == 1 else ("no" if x == 2 else "not sure"))
df['Sterilized'] = df['Sterilized'].apply(lambda x: "yes" if x == 1 else ("no" if x == 2 else "not sure"))
df['FurLength'] = df['FurLength'].apply(lambda x: "short" if x == 1 else 
                                                       ("medium" if x == 2 else 
                                                       ("long" if x == 3 else "not specified")))
df['MaturitySize'] = df['MaturitySize'].apply(lambda x: "small" if x == 1 else 
                                                               ("medium" if x == 2 else 
                                                               ("large" if x == 3 else 
                                                               ("extra large" if x == 4 else "not specified"))))

df['FeeCat'] = df['Fee'].apply(lambda x: "0" if x == 0 else 
                                        ("[1,25)" if x >= 1 and x < 25 else 
                                        ("[25,59)" if x >= 25 and x < 59 else 
                                        ("[59,108)" if x >= 59 and x < 108 else 
                                        ("[108,210)" if x >= 108 and x < 210 else
                                         "[210,3000]")))))

df['DescLength'] = df['Description'].apply(lambda x: len(str(x).split()))
df['HasDesc'] = 'Yes'
df.loc[df.Description.isnull(), 'HasDesc'] = 'No'

df.loc[df['RescuerID'].value_counts()[df['RescuerID']].values < 10, 'RescuerID'] = "Other"

df['Health'] = df['Health'].apply(lambda x: 'Healthy' if x == 1 else 
                                           ('Minor injury' if x == 2 else 
                                           ('Serious injury' if x == 3 else 'Not Sure')))

df['AgeCat'] = df['Age'].apply(lambda x: "[0,6)" if x < 6 else
                                         "[6,12)" if x >= 6 and x < 12 else
                                         "[12,36)" if x >= 12 and x < 36 else
                                         "[36,60)" if x >= 36 and x < 60 else
                                         "[60,96)" if x >= 60 and x < 96 else
                                         "96+")

df = pd.concat([df, imgFeatures], axis=1)
df = pd.concat([df, dfSentiment], axis=1)

if os.path.exists(x+'_prep.csv'):
  os.remove(x+'_prep.csv')
df.to_csv(x+'_prep.csv', index=False)