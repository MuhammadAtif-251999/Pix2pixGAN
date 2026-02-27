from numpy import zeros
from numpy import ones
from numpy.random import randint
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model
from tensorflow.keras.layers import Input 
from keras.layers import Conv2D, Conv2DTranspose, LeakyReLU, Activation, Concatenate, Dropout, Conv2D, BatchNormalization
from matplotlib import pyplot as plt
from tensorflow.keras.utils import plot_model
from os import listdir
from numpy import asarray, load
from numpy import vstack
from tensorflow.keras.preprocessing.image import img_to_array,load_img
from numpy import savez_compressed
from matplotlib import pyplot
import numpy as np
import tensorflow as tf
from keras.layers import Input, Conv2D, Lambda, Concatenate, Activation,SeparableConv2D
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Concatenate, LeakyReLU, Input
from tensorflow.keras.models import Model

def define_generator(image_shape):
    in_image = Input(shape=image_shape)
    channel=64 
    kernel_size=3
    Relu = LeakyReLU()
# First autoEncoder
    E1 = Conv2D(channel, kernel_size, padding='same', dilation_rate=2)(in_image)
    
    E1_1 = Conv2D(channel, 2, strides=2, padding='same')(E1)
    E1_2 = Conv2D(channel, 2, strides=2, padding='same')(E1_1)
    E1_3 = Conv2D(channel, 2, strides=2, padding='same')(E1_2)
    x = Conv2D(channel, 2, padding='same')(E1_3)
    D1_1 = Conv2DTranspose(channel, 2, strides=2, padding='valid')(Concatenate()([ x, E1_3]))  
    D1_2 = Conv2DTranspose(channel, 2, strides=2, padding='valid')(Concatenate()([D1_1, E1_2]))
    D1_3 = Conv2DTranspose(channel, 2, strides=2, padding='valid')(Concatenate()([D1_2, E1_1]))
    
    
# Second autoEncoder
    E2 = Conv2D(channel, kernel_size, padding='same', dilation_rate=2)(in_image)
    
    E2_1 = Conv2D(channel, 2, strides=2, padding='same')(E2)
    E2_2 = Conv2D(channel, 2, strides=2, padding='same')(E2_1)
    E2_3 = Conv2D(channel, 2, strides=2, padding='same')(E2_2)
    x = Conv2D(channel, 2, padding='same')(E2_3)
    D2_1 = Conv2DTranspose(channel, 2, strides=2, padding='valid')(Concatenate()([x ,E2_3]))  
    D2_2 = Conv2DTranspose(channel, 2, strides=2, padding='valid')(Concatenate()([D2_1, E2_2]))
    D2_3 = Conv2DTranspose(channel, 2, strides=2, padding='valid')(Concatenate()([D2_2, E2_1]))
    
    
# Third autoEncoder
    E3 = Conv2D(channel, kernel_size, padding='same', dilation_rate=2)(in_image)
    
    E3_1 = Conv2D(channel, 2, strides=2, padding='same')(Concatenate()([E3 ,D2_3, D1_3 ]))
    E3_2 = Conv2D(channel, 2, strides=2, padding='same')(Concatenate()([E3_1 ,D2_2, D1_2 ]))
    E3_3 = Conv2D(channel, 2, strides=2, padding='same')(Concatenate()([E3_2 ,D2_1, D1_1 ]))
    
    D3_1 = Conv2DTranspose(channel, 2, strides=2, padding='valid')(Concatenate()([E3_3 ,E2_3, E1_3 ]))  
    D3_2 = Conv2DTranspose(channel, 2, strides=2, padding='valid')(Concatenate()([D3_1 ,E2_2, E1_2 ]))
    D3_3 = Conv2DTranspose(channel, 2, strides=2, padding='valid')(Concatenate()([D3_2 ,E2_1, E1_1 ]))
    
    x = Conv2D(channel, kernel_size, padding='same', activation=Relu)(D3_3)
    out_image = Conv2D(3, 1, padding='same')(x)
    
    model = Model(in_image, out_image)
    
    return model


g_model = define_generator((400,600,3))
# plot_model(g_model, to_file = 'gen_model.png', show_shapes=True)