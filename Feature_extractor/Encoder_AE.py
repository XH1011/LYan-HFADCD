# pip install tensorflow_addons
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import h5py
import numpy as np
import math
import os
import random
from scipy import linalg
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array, as_float_array
from Utils import *
from Unet_Utils import *
from tensorflow.python.framework import ops

# Global Settings
batch_size=2
num_epochs=1000
learning_rate=1e-5
img_size=3072
img_channels=1

faults= ['x0','x1','x2','x3','x4','x5','x6']
# faults = ['x0']
root = './Results/data_generate/generate_'
for fault in faults:
    # Build graph
    ops.reset_default_graph()
    # Build encoder
    inputs_=layers.Input(shape=(img_size, img_channels), name="image_input")
    # 2，神经网络
    layers = tf.keras.layers
    # ### Encoder
    conv1 = layers.Conv1D(filters=32, kernel_size=5, padding='same', activation=tf.nn.relu)(inputs_)
    maxpool1 = layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(conv1)
    conv2 = layers.Conv1D(filters=16, kernel_size=5, padding='same', activation=tf.nn.relu)(maxpool1)
    maxpool2 = layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(conv2)
    conv3 = layers.Conv1D(filters=8, kernel_size=5, padding='same', activation=tf.nn.relu)(maxpool2)
    maxpool3 = layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(conv3)
    conv4 = layers.Conv1D(filters=4, kernel_size=5, padding='same', activation=tf.nn.relu)(maxpool3)
    maxpool4 = layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(conv4)
    conv5 = layers.Conv1D(filters=2, kernel_size=5, padding='same', activation=tf.nn.relu)(maxpool4)
    maxpool5 = layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(conv5)
    re = tf.reshape(maxpool5, [-1, 192])
    # -----------#
    # latent = layers.Dense(units=128, activation=tf.nn.relu)(re)
    latent = layers.Dense(units=128)(re)
    # -----------#
    # ---Decoder---#
    x = layers.Dense(units=192, activation=tf.nn.relu)(re)
    x = tf.reshape(x, [-1, 96, 2])
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(filters=4, kernel_size=5, padding='same', activation=tf.nn.relu)(x)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(filters=8, kernel_size=5, padding='same', activation=tf.nn.relu)(x)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(filters=16, kernel_size=5, padding='same', activation=tf.nn.relu)(x)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(filters=32, kernel_size=5, padding='same', activation=tf.nn.relu)(x)
    x = layers.UpSampling1D(2)(x)
    rx = layers.Conv1D(filters=1, kernel_size=5, padding='same', activation=tf.nn.relu)(x)
    # print(rx.shape, inputs_.shape)
    # print('Built Encoder../')

    # print(image_input.shape, enout.shape, x_out.shape)
    # #Build model
    dcae=keras.Model(inputs_, rx)
    #
    # # Opimizer and loss function
    opt = keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-8)
    print('Network Summary-->')
    # dcae.summary()

    # dcae=keras.Model(image_input, x_out, name="new_dcae")
    dir = './Results/model_ae_norelu/model_last_999.ckpt'
    print('Load weights from ', dir)
    dcae.load_weights(dir)
    new_enout=tf.keras.models.Model(inputs=inputs_,outputs=latent)
    # new_enout = tf.keras.models.Model(inputs=dcae.input, outputs=dcae.output)

#data_test
    file_name = root + fault + '_test.pkl'
    # x = LoadData_pickle(file_name)[0]
    x = pickle.load(open(file_name, 'rb'))[0]
    data = tf.reshape(x, shape=[-1, 3072, 1])
    # x_hat1=new_enout.predict(data)
    # x_hat2 = dcae.call(data).numpy()
    extracted_features = new_enout.predict(data)
    print(extracted_features.shape)
    with open('./Results/data_encoded_test/en_'+fault+'_test.pkl', 'wb') as f:
        pickle.dump(extracted_features, f, pickle.HIGHEST_PROTOCOL)

#data_train
    file_name = root + fault + '_train.pkl'
    data = pickle.load(open(file_name, 'rb'))[0]
    # file_name=root+fault+'_train'
    # data=LoadData_pickle(file_name)[0]
    data=tf.reshape(data, shape=[-1, 3072, 1])
    extracted_features = new_enout.predict(data)
    print(extracted_features.shape)
    with open('./Results/data_encoded_test/en_'+fault+'_train.pkl', 'wb') as f:
        pickle.dump(extracted_features, f, pickle.HIGHEST_PROTOCOL)
#

# print(x_hat.shape,x.shape)
#
# plt.figure(1)
# plt.plot(x[1,:].T)
# plt.show()

# x_hat1=np.squeeze(x_hat1)
# plt.figure(2)
# plt.plot(x_hat1[2,:].T,'b')
# plt.show()
#
# x_hat1=np.squeeze(data)
# plt.figure(2)
# plt.plot(x_hat1[1,:].T,'r')
# plt.show()

# plt.figure(3)
# plt.plot(x_hat2[1,:].T)
# plt.show()
#



