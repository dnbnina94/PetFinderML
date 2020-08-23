import tensorflow as tf
import pandas as pd
import os
import math
clear = lambda: os.system('cls')

dftrain = pd.read_csv('../train.csv')
y_train = dftrain.pop('AdoptionSpeed')

dftrain['AgeCat'] = dftrain['Age'].apply(lambda x: "[0,6)" if x < 6 else
                                             "[6,12)" if x >= 6 and x < 12 else
                                             "[12,36)" if x >= 12 and x < 36 else
                                             "[36,60)" if x >= 36 and x < 60 else
                                             "[60,96)" if x >= 60 and x < 96 else
                                             "[96,255)");
dftrain["HasName"] = 1
dftrain.loc[dftrain.Name.isnull(), "HasName"] = 0
dftrain.pop("Health")
dftrain.pop("Name")
dftrain.pop("Description")
dftrain.pop("PetID")

CATEGORICAL_COLUMNS = ["RescuerID", 'AgeCat']
NUMERIC_COLUMNS = ['Type', 'HasName', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2', 'Color3',
                   'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed', 'Sterilized',
                   'Quantity', 'Fee', 'State', 'VideoAmt', 'PhotoAmt']

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
  vocabulary = dftrain[feature_name].unique() 
  feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
  feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

def input_function(data_df, label_df=None, num_epochs=10, shuffle=True, batch_size=32):
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df)) 
    if shuffle:
      ds = ds.shuffle(1000) 
    ds = ds.batch(batch_size).repeat(num_epochs) 
    return ds

linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns, n_classes=5)

# TRAIN
linear_est.train(lambda: input_function(dftrain, y_train))
# EVALUATE
result = linear_est.evaluate(lambda: input_function(dftrain, y_train, num_epochs=1, shuffle=False))

clear() 
print ('Linear Classifier Test Set Accuracy:', result['accuracy'])




