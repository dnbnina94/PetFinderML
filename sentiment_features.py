import pandas as pd
import math
import os
import json

print('Enter file name:')
x = input()

train = pd.read_csv(x+'.csv')
petIds = train['PetID'].values

sentimentFeatures = {}
for (i,petId) in enumerate(petIds):
    try:
        with open('../'+x+'_sentiment/'+petId+'.json') as json_file:
            jsonFile = json.load(json_file)
    except:
        pass
    score = jsonFile['documentSentiment']['score']
    magnitude = jsonFile['documentSentiment']['magnitude']
    language = jsonFile['language']
    sentimentFeatures[petId] = [score, magnitude, language]

dfSentiment = pd.DataFrame.from_dict(sentimentFeatures, orient='index')
dfSentiment.columns = ['Score','Magnitude','Language']
dfSentiment.to_csv(x+'_sentiment_features.csv')
dfSentiment.head()