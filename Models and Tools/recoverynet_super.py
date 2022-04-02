
import tensorflow as tf
import os
from matplotlib import pyplot as plt

from os import listdir
from os.path import isfile, join

import numpy as np
import scipy.io
import pandas as pd
from scipy.io import loadmat
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import Sequential
from tensorflow import print as ptf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from Models_and_Tools.psf_layer_coded_aperture_t1 import *

def Psf_show(input_size=(512, 512, 3), depth=25, depth_out=25,diam=3e-6):

    # define the model input

    inputs = Input(shape=input_size)
    # Customer _layer_psfs
    #ptf("aca")
    conv_psf = Psf_layer_mostrar([input_size[0], input_size[1], depth], distance=50e-3, patch_size=250, sample_interval=diam,
                         wave_resolution = 1000, distance_code = 47e-3, Nt = 5, fac_m = 1, wave_lengths=np.linspace(420, 660, depth)*1e-9, nterms=37, height_tolerance = 20e-9)(inputs)
    model = Model(inputs, conv_psf)
    return model

def Proposed_net(input_size=(512, 512, 3), depth=25, depth_out=25,diam=3e-6):

    # define the model input

    inputs = Input(shape=input_size)
    # Customer _layer_psfs
    #ptf("aca")
    conv_psf = Psf_layer([input_size[0], input_size[1], depth], distance=50e-3, patch_size=250, sample_interval=diam,
                         wave_resolution = 1000, distance_code = 40e-3, Nt = 5, fac_m = 1, wave_lengths=np.linspace(420, 660, depth)*1e-9, nterms=37, height_tolerance = 20e-9)(inputs)



    conv1 = Conv2D(16, (3, 3), padding="same", kernel_initializer='he_normal', activation="relu")(conv_psf)
    conv1 = Dropout(0.1)(conv1)
    conv1 = Conv2D(16, (3, 3), padding="same", kernel_initializer='he_normal', activation="relu",
                   dilation_rate=(2, 2))(conv1)

    #125 x 125
    conv11 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv12 = Conv2D(32, (3, 3), padding="same", kernel_initializer='he_normal', activation="relu")(conv11)
    conv12 = Dropout(0.1)(conv12)
    conv12 = Conv2D(32, (3, 3), padding="same", kernel_initializer='he_normal', activation="relu",
                    dilation_rate=(2, 2))(conv12)

    #25x25
    conv15 = MaxPooling2D(pool_size=(5, 5))(conv12)
    conv15 = Conv2D(64, (3, 3), padding="same", kernel_initializer='he_normal', activation="relu")(conv15)
    conv15 = Dropout(0.2)(conv15)
    conv15 = Conv2D(64, (3, 3), padding="same", kernel_initializer='he_normal', activation="relu",
                    dilation_rate=(2, 2))(conv15)

    #125x125
    conv13 = UpSampling2D(size=(5, 5))(conv15)

    # 20 + 28 bands (125x125)
    conv1312 = Concatenate()([conv12, conv13])

    conv151 = Conv2D(32, (3, 3), padding="same", kernel_initializer='he_normal', activation="relu")(conv1312)
    conv151 = Dropout(0.2)(conv151)
    conv151 = Conv2D(32, (3, 3), padding="same", kernel_initializer='he_normal', activation="relu",
                    dilation_rate=(2, 2))(conv151)

    # 250x250
    conv1250 = UpSampling2D(size=(2, 2))(conv151)

    conv250 = Concatenate()([conv1250, conv1])

    conv250 = Conv2D(32, (3, 3), padding="same", kernel_initializer='he_normal', activation="relu")(conv250)
    conv250 = Dropout(0.1)(conv250)
    conv250 = Conv2D(32, (3, 3), padding="same", kernel_initializer='he_normal', activation="relu",
                     dilation_rate=(2, 2))(conv250)

    #500x500x36
    #conv500 = UpSampling2D(size=(2, 2))(conv250)

    conv500 = Conv2D(32, (3, 3), padding="same", kernel_initializer='he_normal', activation="relu")(conv250)
    conv500 = Conv2D(32, (3, 3), padding="same", kernel_initializer='he_normal', activation="relu",
                     dilation_rate=(2, 2))(conv500)

    conv5 = Conv2D(32, (1, 1), padding="same", kernel_initializer='he_normal')(conv500)

    conv6 = Conv2D(32, (1, 1), padding="same", kernel_initializer='he_normal')(conv5)

    # CONV => RELU => BN
    final = Conv2D(depth_out, (1, 1), padding="same", kernel_initializer='he_normal')(conv6)


    # construct the CNN
    model = Model(inputs, final)

    return model
