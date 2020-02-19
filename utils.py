import numpy as np
import keras
from random import shuffle
import keras.backend as K
from PIL import Image
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from numpy.random import rand
from keras.applications import *
from keras.layers import *
from keras.models import Model
import random
import tensorflow as tf
import os
import pandas as pd


def model_inception_multigap(input_shape=(None,None,3)):
    model_base = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
    model_base.load_weights('save_model/InceptionV3_4jnd.hdf5')
    feature_layers = [model_base.get_layer('mixed%d' % i) for i in range(11)]
    gaps = [GlobalAveragePooling2D(name="gap%d"%i)(l.output) for i,l in enumerate(feature_layers)]
    concat_gaps = Concatenate(name='concatenated_gaps')(gaps)
    return Model(inputs=model_base.input, outputs=concat_gaps)
   
       
def fc_model():
    
    input1 = Input(shape=(10048,))
    input2 = Input(shape=(10048,))
    tmp = subtract([input1, input2])
    
    
    out = Concatenate()([input1, tmp, input2])
    out = Dropout(0.25)(out)
    out = Dense(1024, kernel_initializer='he_normal', activation='relu')(out)
    out = BatchNormalization()(out)
    out = Dropout(0.25)(out)
    out = Dense(512, kernel_initializer='he_normal', activation='relu')(out)
    out = BatchNormalization()(out)
    out = Dropout(0.25)(out)
    out = Dense(128, kernel_initializer='he_normal', activation='relu')(out)
    out = BatchNormalization()(out)
    out = Dropout(0.5)(out)
    output = Dense(1, activation='linear', name='sur')(out)

    model = Model([input1,input2],output)
    
    return model    
    
    
    
    
    