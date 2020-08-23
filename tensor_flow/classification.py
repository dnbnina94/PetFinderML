import tensorflow as tf
import pandas as pd
import os
import math

clear = lambda: os.system('cls')

dftrain = pd.read_csv('../train.csv')
y_train = dftrain.pop('AdoptionSpeed')

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

CATEGORICAL_COLUMNS = ["RescuerID"]
NUMERIC_COLUMNS = ['Type', 'HasName', 'AgeCat', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2', 'Color3',
                   'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed', 'Sterilized',
                   'Quantity', 'Fee', 'State', 'HasVideos', 'HasPhotos']

def input_fn(features, labels, training=True, batch_size=256):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    if training:
        dataset = dataset.shuffle(1000).repeat()
    return dataset.batch(batch_size)

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
  vocabulary = dftrain[feature_name].unique() 
  feature_columns.append(tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary)))

for feature_name in NUMERIC_COLUMNS:
  feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[30, 10],
    n_classes=5)

classifier.train(
    input_fn=lambda: input_fn(dftrain, y_train, training=True),
    steps=5000)

eval_result = classifier.evaluate(
    input_fn=lambda: input_fn(dftrain, y_train, training=False))

clear()
print('\nDeep Neural Network Test Set Accuracy: {accuracy:0.3f}\n'.format(**eval_result))

