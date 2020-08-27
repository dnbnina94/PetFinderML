import cv2
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.applications.densenet import preprocess_input, DenseNet121
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Input, Lambda, AveragePooling1D
import keras.backend as K
import tensorflow as tf
config = tf.compat.v1.ConfigProto(gpu_options = 
                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
# device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)
# tf.compat.v1.disable_eager_execution()

# config = tf.compat.v1.ConfigProto(
#         device_count = {'GPU': 0}
#     )
# sess = tf.compat.v1.Session(config=config)

def resize_to_square(im):
    oldSize = im.shape[:2]
    ratio = float(imgSize)/max(oldSize)
    newSize = tuple([int(x*ratio) for x in oldSize])
    im = cv2.resize(im, (newSize[1], newSize[0]))
    deltaW = imgSize - newSize[1]
    deltaH = imgSize - newSize[0]
    top, bottom = deltaH//2, deltaH-(deltaH//2)
    left, right = deltaW//2, deltaW-(deltaW//2)
    color = [0, 0, 0]
    newIm = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,value=color)
    return newIm

def load_image(pet_id):
    image = cv2.imread('../train_img/'+pet_id+'-1.jpg')
    newImage = resize_to_square(image)
    newImage = preprocess_input(newImage)
    return newImage

dftrain = pd.read_csv('train.csv')
imgSize = 256
batchSize = 16

petIds = dftrain['PetID'].values
numOfBatches = len(petIds) // batchSize + 1

inp = Input((256,256,3))
backbone = DenseNet121(input_tensor = inp, include_top = False)
x = backbone.output
x = GlobalAveragePooling2D()(x)
x = Lambda(lambda x: K.expand_dims(x,axis = -1))(x)
x = AveragePooling1D(4)(x)
out = Lambda(lambda x: x[:,:,0])(x)

m = Model(inp,out)

features = {}
for b in range(numOfBatches):
    start = b*batchSize
    end = (b+1)*batchSize
    batchPets = petIds[start:end]
    batchImages = np.zeros((len(batchPets),imgSize,imgSize,3))
    for i,petId in enumerate(batchPets):
        try:
            batchImages[i] = load_image(petId)
        except:
        	pass
    batchPreds = m.predict(batchImages)
    for i,petId in enumerate(batchPets):
        features[petId] = batchPreds[i]

trainFeats = pd.DataFrame.from_dict(features, orient='index')

trainFeats.to_csv('train_img_features.csv')
trainFeats.head()
