
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
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from psf_layer_coded_aperture_t3 import *

def Proposed_net(pretrained_weights=None, input_size=(512, 512, 3), depth=24):

    # define the model input

    inputs = Input(shape=input_size)
    # Customer _layer_psfs
    conv_psf = Psf_layer((input_size[0],input_size[1],depth), distance=50e-3, patch_size=250, sample_interval=3.69e-6,
    wave_resolution = 500,distance_code = 48e-3, Nt = 5, fac_m = 1, wave_lengths=np.linspace(420, 660, depth)*1e-9,nterms = 37, height_tolerance = 20e-9)(inputs)

     #conv_psf = Psf_layer((input_size[0],input_size[1],depth), distance=50e-3, patch_size=512, sample_interval=3.69e-6,
     #discrete_size_sensor=(0.007032/256),patch_len=64, wave_lengths=np.linspace(420, 660, depth)*1e-9,nterms = 230, height_tolerance = None)(inputs)

    conv1 = Conv2D(6 * depth, (3, 3), padding="same", kernel_initializer='he_normal')(conv_psf)
    conv1 = Activation("relu")(conv1)

    # CONV => RELU => BN
    conv2 = Conv2D(9 * depth, (5, 5), padding="same", kernel_initializer='he_normal')(conv1)

    convi = Conv2D(6 * depth, (3, 3), padding="same", kernel_initializer='he_normal')(conv2)

    # CONV => RELU => BN
    ##conv3 = Conv2D(depth, (3, 3), padding="same", kernel_initializer='he_normal')(convi)
    conv3 = Conv2D(3, (3, 3), padding="same", kernel_initializer='he_normal')(convi)

    # CONV3 + inputs
    conv4 = Add()([conv3, conv_psf])

    conv5 = Conv2D(6 * depth, (1, 1), padding="same", kernel_initializer='he_normal')(conv4)

    conv6 = Conv2D(3 * depth, (1, 1), padding="same", kernel_initializer='he_normal')(conv5)

    # CONV => RELU => BN
    final = Conv2D(depth, (1, 1), padding="same", kernel_initializer='he_normal')(conv6)


    # construct the CNN
    model = Model(inputs, final)

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model