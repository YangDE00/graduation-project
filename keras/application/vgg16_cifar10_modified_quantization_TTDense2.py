import keras  
from keras.datasets import cifar10  
from keras.preprocessing.image import ImageDataGenerator  
from keras.models import Sequential  
from keras.layers import Dense, Dropout, Activation, Flatten  
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization  
from keras import optimizers  
import numpy as np  
from keras.layers.core import Lambda  
from keras import backend as K  
from keras.optimizers import SGD  
from keras import regularizers  
from keras.initializers import he_normal
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, TensorBoard

from keras.utils import np_utils
import os
import t3f
from utils.tt_dense import TTDense
from layers import *
from utils import *
from utils.tt_dense import TTDense
from quantized_ops import quantized_tanh as quantized_tanh_op
from quantized_layers import QuantizedConv2D, QuantizedDense

def quantized_tanh(x):
    return quantized_tanh_op(x) 

weight_decay = 0.0001  
epochs       = 200
iterations   = 391 
batch_size=128
dropout= 0.5
num_classes = 10
log_filepath = r'./vgg19_retrain_logs/'

kernel_size=(3,3)
use_bias=False
H=1.
nb=8
kernel_lr_multiplier='Glorot'

def scheduler(epoch):
    if epoch < 80:
        return 0.1
    if epoch < 160:
        return 0.01
    return 0.001


# data loading
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# data preprocessing 
x_train[:,:,:,0] = (x_train[:,:,:,0]-123.680)
x_train[:,:,:,1] = (x_train[:,:,:,1]-116.779)
x_train[:,:,:,2] = (x_train[:,:,:,2]-103.939)
x_test[:,:,:,0] = (x_test[:,:,:,0]-123.680)
x_test[:,:,:,1] = (x_test[:,:,:,1]-116.779)
x_test[:,:,:,2] = (x_test[:,:,:,2]-103.939)

# build model
model = Sequential()

# Block 1
model.add(QuantizedConv2D(64, kernel_size=kernel_size,data_format='channels_last',
H=H, nb=nb, kernel_lr_multiplier=kernel_lr_multiplier, 
kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(),
                       padding='same', use_bias=use_bias,name='block1_conv1', input_shape=(32,32,3))) 
model.add(BatchNormalization())
model.add(Activation(quantized_tanh))
model.add(QuantizedConv2D(64, kernel_size=kernel_size,data_format='channels_last',
H=H, nb=nb, kernel_lr_multiplier=kernel_lr_multiplier, 
kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(),
                       padding='same', use_bias=use_bias, name='block1_conv2'))
model.add(BatchNormalization())
model.add(Activation(quantized_tanh))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

# Block 2
model.add(QuantizedConv2D(128, kernel_size=kernel_size,data_format='channels_last',
H=H, nb=nb, kernel_lr_multiplier=kernel_lr_multiplier, 
kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(),
                       padding='same', use_bias=use_bias, name='block2_conv1'))
model.add(BatchNormalization())
model.add(Activation(quantized_tanh))
model.add(QuantizedConv2D(128, kernel_size=kernel_size,data_format='channels_last',
H=H, nb=nb, kernel_lr_multiplier=kernel_lr_multiplier, 
kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(),
                       padding='same', use_bias=use_bias, name='block2_conv2'))
model.add(BatchNormalization())
model.add(Activation(quantized_tanh))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

# Block 3
model.add(QuantizedConv2D(256, kernel_size=kernel_size,data_format='channels_last',
H=H, nb=nb, kernel_lr_multiplier=kernel_lr_multiplier, 
kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(),
                       padding='same', use_bias=use_bias, name='block3_conv1'))
model.add(BatchNormalization())
model.add(Activation(quantized_tanh))
model.add(QuantizedConv2D(256, kernel_size=kernel_size,data_format='channels_last',
H=H, nb=nb, kernel_lr_multiplier=kernel_lr_multiplier, 
kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(),
                       padding='same', use_bias=use_bias,name='block3_conv2'))
model.add(BatchNormalization())
model.add(Activation(quantized_tanh))
model.add(QuantizedConv2D(256, kernel_size=kernel_size,data_format='channels_last',
H=H, nb=nb, kernel_lr_multiplier=kernel_lr_multiplier, 
kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(),
                       padding='same', use_bias=use_bias, name='block3_conv3'))
model.add(BatchNormalization())
model.add(Activation(quantized_tanh))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

# Block 4
model.add(QuantizedConv2D(512, kernel_size=kernel_size,data_format='channels_last',
H=H, nb=nb, kernel_lr_multiplier=kernel_lr_multiplier, 
kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(),
                       padding='same', use_bias=use_bias, name='block4_conv1'))
model.add(BatchNormalization())
model.add(Activation(quantized_tanh))
model.add(QuantizedConv2D(512, kernel_size=kernel_size,data_format='channels_last',
H=H, nb=nb, kernel_lr_multiplier=kernel_lr_multiplier, 
kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(),
                       padding='same', use_bias=use_bias,name='block4_conv2'))
model.add(BatchNormalization())
model.add(Activation(quantized_tanh))
model.add(QuantizedConv2D(512, kernel_size=kernel_size,data_format='channels_last',
H=H, nb=nb, kernel_lr_multiplier=kernel_lr_multiplier, 
kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(),
                       padding='same', use_bias=use_bias, name='block4_conv3'))
model.add(BatchNormalization())
model.add(Activation(quantized_tanh))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

# Block 5
model.add(QuantizedConv2D(512, kernel_size=kernel_size,data_format='channels_last',
H=H, nb=nb, kernel_lr_multiplier=kernel_lr_multiplier, 
kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(),
                       padding='same', use_bias=use_bias,name='block5_conv1'))
model.add(BatchNormalization())
model.add(Activation(quantized_tanh))
model.add(QuantizedConv2D(512, kernel_size=kernel_size,data_format='channels_last',
H=H, nb=nb, kernel_lr_multiplier=kernel_lr_multiplier, 
kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(),
                       padding='same', use_bias=use_bias,name='block5_conv2'))
model.add(BatchNormalization())
model.add(Activation(quantized_tanh))
model.add(QuantizedConv2D(512, kernel_size=kernel_size,data_format='channels_last',
H=H, nb=nb, kernel_lr_multiplier=kernel_lr_multiplier, 
kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(),
                       padding='same', use_bias=use_bias, name='block5_conv3'))
model.add(BatchNormalization())
model.add(Activation(quantized_tanh))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))

# model modification for cifar-10
model.add(Flatten(name='flatten'))
model.add(TTDense(row_dims=[4,4,8,4],column_dims=[8,8,8,8],
         tt_rank=4,activation='relu',bias_init=1e-3,bias=True, 
         init='he', name='fc1'))

model.add(BatchNormalization())
model.add(Dropout(dropout))

model.add(TTDense(row_dims=[8,8,8,8],column_dims=[8,8,8,8],
         tt_rank=4,activation='relu',bias_init=1e-3,bias=True, 
         init='he', name='fc2'))  
model.add(BatchNormalization())
model.add(Dropout(dropout))   
   
model.add(Dense(10, kernel_regularizer=keras.regularizers.l2(weight_decay),
          kernel_initializer=he_normal(), name='predictions_cifa10'))        
model.add(BatchNormalization())
model.add(Activation('softmax'))
 
model.summary()
  
sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

tb_cb = TensorBoard(log_dir=log_filepath, histogram_freq=0)
change_lr = LearningRateScheduler(scheduler)
cbks = [change_lr,tb_cb]

print('Using real-time data augmentation.')
datagen = ImageDataGenerator(horizontal_flip=True,
        width_shift_range=0.125,height_shift_range=0.125,fill_mode='constant',cval=0.)

datagen.fit(x_train)

model.fit_generator(datagen.flow(x_train, y_train,batch_size=batch_size),
                    steps_per_epoch=iterations,
                    epochs=epochs,
                    callbacks=cbks,
                    validation_data=(x_test, y_test))