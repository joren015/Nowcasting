from numpy.core.defchararray import index
from tensorflow import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
#from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Input, ConvLSTM2D, BatchNormalization
from tensorflow.keras.layers import Conv3D, MaxPool3D, Conv3DTranspose, Add
from tensorflow.keras.layers import SpatialDropout3D, UpSampling3D, Dropout, RepeatVector, Average
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import mse, mae, Huber
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from typing import Any, Tuple
mpl.rcParams['figure.dpi'] = 300
import argparse
import json
#from sklearn.linear_model import LinearRegression
#from sklearn.ensemble import RandomForestRegressor
import tensorflow.keras.layers as layers
from keras.layers import (Activation, BatchNormalization, Conv3D,
                          Conv3DTranspose, ConvLSTM2D, Cropping3D, Dropout,
                          Input, MaxPooling3D, ZeroPadding3D, concatenate)

def unet_conv3d(input_shape: Tuple[int] = (12, 256, 620, 4),
         num_filters_base: int = 8,
         dropout_rate: float = 0.2):
    
    input_shape = (12,256,620,4)
    inputs = Input(shape=input_shape)
    x_init = BatchNormalization()(inputs)  # Try with normalizing the dataset
    x0 = ZeroPadding3D(padding=(0, 0, 2))(x_init)
    
        
    x_conv1_b1 = Conv3D(filters= num_filters_base, kernel_size=(1,3,3), padding='same', activation="relu")(x0)
    x_conv2_b1 = Conv3D(filters=num_filters_base, kernel_size=(1,3,3), padding='same', activation="relu")(x_conv1_b1)
    x_max_b1 = MaxPool3D([1, 2, 2],padding='same')(x_conv2_b1)
    x_bn_b1 = BatchNormalization()(x_max_b1)
    x_do_b1 = Dropout(dropout_rate)(x_bn_b1)
    
    x_conv1_b2 = Conv3D(filters=2*num_filters_base, kernel_size=(1,3,3), padding='same', activation="relu")(x_do_b1)
    x_conv2_b2 = Conv3D(filters=2*num_filters_base, kernel_size=(1,3,3), padding='same', activation="relu")(x_conv1_b2)
    x_max_b2 = MaxPool3D([1, 2, 2],padding='same')(x_conv2_b2)
    x_bn_b2 = BatchNormalization()(x_max_b2)
    x_do_b2 = Dropout(dropout_rate)(x_bn_b2)
    
    x_conv1_b3 = Conv3D(filters=4*num_filters_base, kernel_size=(1,3,3), padding='same', activation="relu")(x_do_b2)
    x_conv2_b3 = Conv3D(filters=4*num_filters_base, kernel_size=(1,3,3), padding='same', activation="relu")(x_conv1_b3)
    x_max_b3 = MaxPool3D([1, 2, 2],padding='same')(x_conv2_b3)
    x_bn_b3 = BatchNormalization()(x_max_b3)
    x_do_b3 = Dropout(dropout_rate)(x_bn_b3)
    
    x_conv1_b4 = Conv3D(filters=8*num_filters_base, kernel_size=(1,3,3), padding='same', activation="relu")(x_do_b3)
    x_conv2_b4 =Conv3D(filters=8*num_filters_base, kernel_size=(1,3,3), padding='same', activation="relu")(x_conv1_b4)
    x_max_b4 = MaxPool3D([1, 2, 2],padding='same')(x_conv2_b4)
    x_bn_b4 = BatchNormalization()(x_max_b4)
    x_do_b4 = Dropout(dropout_rate)(x_bn_b4)
    
    # ------- Head Residual Output (Residual Decoder)
    
    x_conv1_b5 = Conv3D(filters=8*num_filters_base, kernel_size=(1,3,3), padding='same', activation="relu")(x_do_b4)
    x_conv2_b5 =Conv3D(filters=8*num_filters_base, kernel_size=(1,3,3), padding='same', activation="relu")(x_conv1_b5)
    x_deconv_b5 = Conv3DTranspose(filters=8*num_filters_base, kernel_size=(1, 2, 2),strides=(1,2,2),padding='same', activation="relu")(x_conv2_b5)
    x_bn_b5 = BatchNormalization()(x_deconv_b5)
    x_do_b5 = Dropout(dropout_rate)(x_bn_b5)
    
    
    cropped_x_conv2_b4 = layers.Cropping3D(cropping=((11,0),(0,0),(0,0)),data_format="channels_last")(x_conv2_b4)
    cropped_x_conv2_b4 = layers.concatenate([cropped_x_conv2_b4]*12,axis=1)
    x_conv1_b6 = Conv3D(filters=4*num_filters_base, kernel_size=(2,1,1), activation="relu")(layers.concatenate([cropped_x_conv2_b4,x_do_b5]))
    x_conv2_b6 = Conv3D(filters=4*num_filters_base, kernel_size=(1, 1, 1),padding='same', activation="relu")(x_conv1_b6)
    x_deconv_b6 = Conv3DTranspose(filters=4*num_filters_base,  kernel_size=(1, 2, 2),strides = (1,2,2),padding='same', activation="relu")(x_conv2_b6)
    x_bn_b6 = BatchNormalization()(x_deconv_b6)
    x_do_b6 = Dropout(dropout_rate)(x_bn_b6)
    
    
    cropped_x_conv2_b3 = layers.Cropping3D(cropping=((11,0),(0,0),(0,0)),data_format="channels_last")(x_conv2_b3)
    cropped_x_conv2_b3 = layers.concatenate([cropped_x_conv2_b3]*11,axis=1)
    x_conv1_b7 =Conv3D(filters=2*num_filters_base, kernel_size=(2,1,1), activation="relu")(layers.concatenate([cropped_x_conv2_b3,x_do_b6]))
    x_conv2_b7 = Conv3D(filters=2*num_filters_base, kernel_size=(1, 1, 1),padding='same', activation="relu")(x_conv1_b7)
    x_deconv_b7 =  Conv3DTranspose(filters=2*num_filters_base,  kernel_size=(1, 2, 2),strides = (1,2,2),padding='same', activation="relu")(x_conv2_b7)
    x_bn_b7 = BatchNormalization()(x_deconv_b7)
    x_do_b7 = Dropout(dropout_rate)(x_bn_b7)
    
    
    cropped_x_conv2_b2 =  layers.Cropping3D(cropping=((11,0),(0,0),(0,0)),data_format="channels_last")(x_conv2_b2)
    cropped_x_conv2_b2 = layers.concatenate([cropped_x_conv2_b2]*10,axis=1)
    x_conv1_b8 = Conv3D(filters=1*num_filters_base, kernel_size=(2,1,1), activation="relu")(layers.concatenate([cropped_x_conv2_b2,x_do_b7]))
    x_conv2_b8 = Conv3D(filters=1*num_filters_base, kernel_size=(1, 1, 1),padding='same', activation="relu")(x_conv1_b8)
    x_deconv_b8 = Conv3DTranspose(filters=2*num_filters_base,  kernel_size=(1, 2, 2),strides = (1,2,2),padding='same', activation="relu")(x_conv2_b8)
    x_bn_b8 = BatchNormalization()(x_deconv_b8)
    x_do_b8 = Dropout(dropout_rate)(x_bn_b8)
    
    
    cropped_x_conv2_b1 = layers.Cropping3D(cropping=((11,0),(0,0),(0,0)),data_format="channels_last")(x_conv2_b1)
    cropped_x_conv2_b1 = layers.concatenate([cropped_x_conv2_b1]*9,axis=1)
    x_conv1_b9 =Conv3D(filters=0.5*num_filters_base, kernel_size=(2,1,1), activation="relu")(layers.concatenate([cropped_x_conv2_b1,x_do_b8]))
    x_conv2_b9 = Conv3D(filters=0.5*num_filters_base, kernel_size=(1, 1, 1),padding='same', activation="relu")(x_conv1_b9)
    x_bn_b9 = BatchNormalization()(x_conv2_b9)
    x_do_b9 = Dropout(dropout_rate)(x_bn_b9)
    
    
    residual_output = Conv3DTranspose(1, kernel_size=(1, 1, 1), padding="same")(x_do_b9)
    output = Activation("linear", dtype="float32")(residual_output)
    output = Cropping3D(cropping=(0, 0, 2))(output)
    
    output = tf.squeeze(output, axis=4)

    model=Model(inputs, output)
    model.summary()
    return model

