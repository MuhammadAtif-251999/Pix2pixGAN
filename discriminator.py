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

from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Adagrad, Nadam,Adadelta
def define_discriminator(image_shape):
    init = RandomNormal(stddev=0.2)

    in_src_Image = Input(shape=image_shape)
    in_target_Image = Input(shape=image_shape)

    merged = Concatenate()([in_src_Image, in_target_Image])

    d = Conv2D(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(merged)
    d = LeakyReLU(alpha=0.2)(d)

    d = Conv2D(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)

    d = Conv2D(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)

    d = Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)

    d = Conv2D(1, (4, 4), padding='same', kernel_initializer=init)(d)
    patch_out = Activation('sigmoid')(d)

    model = Model([in_src_Image, in_target_Image], patch_out)

    opt = Adadelta(learning_rate=1.0, rho=0.95)
    # RMSprop(learning_rate=0.001)

    model.compile(loss='binary_crossentropy',  optimizer=opt, loss_weights=[0.5])
    return model

d_model = define_discriminator((400, 600, 3))
# plot_model(disc_model, to_file='disc_model.png', show_shapes=True)
