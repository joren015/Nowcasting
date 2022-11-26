from typing import Any, Tuple

import tensorflow as tf
from keras import Input, Model
from keras.layers import (Activation, BatchNormalization, Conv3D,
                          Conv3DTranspose, ConvLSTM2D, Cropping3D, Dropout,
                          Input, MaxPooling3D, ZeroPadding3D, concatenate)
from keras.models import Model


def unet_down(x0: Any, num_filters: int = 3, layer_batch_id: int = 1) -> Any:
    """
    unet_down Dimensionality reduction segment of unet structure

    Set of neural network layers that reduce the dimensionality of an input (part of unet structure)

    Parameters
    ----------
    x0 : Any
        Most recent layer in unet structure
    num_filters : int, optional
        Number of filters to use for this unet segment, by default 3
    layer_batch_id : int, optional
        Identifier used to differenciate unet segments, by default 1

    Returns
    -------
    Any
        Next aggregate layer in unet structure
    """
    x1 = (ConvLSTM2D(filters=num_filters,
                     kernel_size=(3, 3),
                     padding="same",
                     name=f"conv_lstm{layer_batch_id}",
                     activation="relu",
                     return_sequences=True))(x0)
    x2 = (ConvLSTM2D(filters=num_filters,
                     kernel_size=(3, 3),
                     padding="same",
                     name=f"conv_lstm{layer_batch_id}2",
                     activation="relu",
                     return_sequences=True))(x1)
    c1 = (ConvLSTM2D(filters=num_filters,
                     kernel_size=(3, 3),
                     padding="same",
                     name=f"conv_lstm{layer_batch_id}3",
                     activation="relu",
                     return_sequences=True))(x2)

    return c1


def unet_up(x0: Any,
            c0: Any,
            num_filters: int = 3,
            init_kernel_size: Tuple[int] = (2, 1, 1),
            layer_batch_id: int = 1):
    """
    unet_up Dimensionality expansion segment of unet structure

    Set of neural network layers that increase the dimensionality of an input (part of unet structure)

    Parameters
    ----------
    x0 : Any
        Most recent layer in unet structure
    c0 : Any
        Previous layer in unet structure to concatinate onto a later layer
    num_filters : int, optional
        Number of filters to use for this unet segment, by default 3
    init_kernel_size : Tuple[int], optional
        Initial kernel size to use for dimensionality expansion, by default (2, 1, 1)
    layer_batch_id : int, optional
        Identifier used to differenciate unet segments, by default 1

    Returns
    -------
    Any
        Next aggregate layer in unet structure
    """
    x1 = Conv3D(filters=num_filters,
                kernel_size=init_kernel_size,
                padding="valid",
                activation="relu")(x0)
    x2 = (ConvLSTM2D(filters=num_filters,
                     kernel_size=(1, 1),
                     padding="same",
                     name=f"conv_lstm{layer_batch_id}",
                     activation="relu",
                     return_sequences=True))(x1)
    x3 = Conv3DTranspose(filters=num_filters,
                         kernel_size=(1, 2, 2),
                         strides=(1, 2, 2),
                         padding="same",
                         activation="relu")(x2)
    x4 = concatenate([x3, c0])
    x5 = (ConvLSTM2D(filters=num_filters,
                     kernel_size=(3, 3),
                     padding="same",
                     name=f"conv_lstm{layer_batch_id}1",
                     activation="relu",
                     return_sequences=True))(x4)
    x6 = (ConvLSTM2D(filters=num_filters,
                     kernel_size=(3, 3),
                     padding="same",
                     name=f"conv_lstm{layer_batch_id}2",
                     activation="relu",
                     return_sequences=True))(x5)
    c1 = (ConvLSTM2D(filters=num_filters,
                     kernel_size=(3, 3),
                     padding="same",
                     name=f"conv_lstm{layer_batch_id}3",
                     activation="relu",
                     return_sequences=True))(x6)

    return c1


def unet_core(x0: Any, num_filters_base: int = 8, dropout_rate: float = 0.2):
    """
    unet_core Core unet structure

    Common segment of layers used in our unet structure (used in res1 and res2 networks)

    Parameters
    ----------
    x0 : Any
        Initial layer in unet structure
    num_filters_base : int, optional
        Number of filters to use as a base for the unet structure, by default 8
    dropout_rate : float, optional
        Constant dropout rate used for the unet structure, by default 0.2

    Returns
    -------
    Any
        Next aggregate layer in unet structure
    """
    x1 = (ConvLSTM2D(filters=num_filters_base,
                     kernel_size=(3, 3),
                     padding="same",
                     name="conv_lstm1",
                     activation="relu",
                     return_sequences=True))(x0)
    c1 = (ConvLSTM2D(filters=num_filters_base,
                     kernel_size=(3, 3),
                     padding="same",
                     name="conv_lstm13",
                     activation="relu",
                     return_sequences=True))(x1)
    x2 = MaxPooling3D(pool_size=(1, 2, 2))(c1)
    x3 = BatchNormalization(center=True, scale=True)(x2)
    x4 = Dropout(dropout_rate)(x3)

    c2 = unet_down(x4, num_filters=2 * num_filters_base, layer_batch_id=2)
    x7 = MaxPooling3D(pool_size=(1, 2, 2))(c2)
    x8 = BatchNormalization(center=True, scale=True)(x7)
    x9 = Dropout(dropout_rate)(x8)

    c3 = unet_down(x9, num_filters=4 * num_filters_base, layer_batch_id=3)
    x12 = MaxPooling3D(pool_size=(1, 2, 2))(c3)
    x13 = BatchNormalization(center=True, scale=True)(x12)
    x14 = Dropout(dropout_rate)(x13)

    c4 = unet_down(x14, num_filters=8 * num_filters_base, layer_batch_id=4)
    x17 = BatchNormalization(center=True, scale=True)(c4)
    x18 = Dropout(dropout_rate)(x17)

    c5 = unet_up(x18,
                 c3[:, 1:12, :, :, :],
                 num_filters=4 * num_filters_base,
                 init_kernel_size=(2, 1, 1),
                 layer_batch_id=5)
    x26 = BatchNormalization(center=True, scale=True)(c5)
    x27 = Dropout(dropout_rate)(x26)

    c6 = unet_up(x27,
                 c2[:, 2:12, :, :, :],
                 num_filters=2 * num_filters_base,
                 init_kernel_size=(2, 1, 1),
                 layer_batch_id=6)
    x34 = BatchNormalization(center=True, scale=True)(c6)
    x35 = Dropout(dropout_rate)(x34)

    x42 = unet_up(x35,
                  c1[:, 4:12, :, :, :],
                  num_filters=num_filters_base,
                  init_kernel_size=(3, 1, 1),
                  layer_batch_id=7)

    return x42


def res1(input_shape: Tuple[int] = (12, 120, 120, 3),
         num_filters_base: int = 8,
         dropout_rate: float = 0.2):
    """
    res1 Unet structure for square input shape

    Unet structure based on an input that is of shape (12, x, x, 3)

    Parameters
    ----------
    input_shape : Tuple[int], optional
        Shape of input, by default (12, 120, 120, 3)
    num_filters_base : int, optional
        Number of filters to use as a base for the unet structure, by default 8
    dropout_rate : float, optional
        Constant dropout rate used for the unet structure, by default 0.2

    Returns
    -------
    Any
        Unet model
    """
    inputs = Input(shape=input_shape)

    x_init = BatchNormalization()(inputs)  # Try with normalizing the dataset

    x42 = unet_core(x_init,
                    num_filters_base=num_filters_base,
                    dropout_rate=dropout_rate)

    residual_output = Conv3D(1, kernel_size=(1, 1, 1), padding="same")(x42)
    output = Activation("linear", dtype="float32")(residual_output)
    output = tf.squeeze(output, axis=4)

    model = Model(inputs, output)
    return model


def res2(input_shape: Tuple[int] = (12, 256, 620, 4),
         num_filters_base: int = 8,
         dropout_rate: float = 0.2):
    """
    res1 Unet structure for rectangular input shape

    Unet structure based on an input that is of shape (12, 256, 620, 4)

    Parameters
    ----------
    input_shape : Tuple[int], optional
        Shape of input, by default (12, 256, 620, 4)
    num_filters_base : int, optional
        Number of filters to use as a base for the unet structure, by default 8
    dropout_rate : float, optional
        Constant dropout rate used for the unet structure, by default 0.2

    Returns
    -------
    Any
        Unet model
    """
    inputs = Input(shape=input_shape)

    x_init = BatchNormalization()(inputs)  # Try with normalizing the dataset
    x0 = ZeroPadding3D(padding=(0, 0, 2))(x_init)

    x42 = unet_core(x0,
                    num_filters_base=num_filters_base,
                    dropout_rate=dropout_rate)

    residual_output = Conv3D(1, kernel_size=(1, 1, 1), padding="same")(x42)
    output = Activation("linear", dtype="float32")(residual_output)
    output = Cropping3D(cropping=(0, 0, 2))(output)
    output = tf.squeeze(output, axis=4)

    model = Model(inputs, output)
    return model
