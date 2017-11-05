import numpy as np
import sys

import tensorflow as tf
from keras.models import Model
from keras.layers import Input, merge, Dropout, Dense, Flatten
from keras.layers import Activation,LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import MaxPooling2D, Conv2D, AveragePooling2D
from keras import backend as K
from keras import optimizers

from matplotlib import pyplot as plt

from InpGen import InputGenerator

def __main__():
    size = (99,99)
    model = create_model((size[0],size[1],3),4)
    model.summary()

    rms = optimizers.RMSprop(lr =0.1,decay=1e-5)
    sgd  = optimizers.SGD(
            lr=0.001,
            decay = 1e-2,
            momentum=0.8,
            nesterov=True)

    print "Compiling model"
    model.compile(loss='mean_squared_error',
                optimizer=sgd,
                metrics=['mae'])

    gen = InputGenerator(size=size, batch_size = 64)
    im,mask,params = next(gen)
    print params[:8]
    for i in range(8):
        plt.subplot(281+i)
        plt.imshow(im[i])
        plt.subplot(2,8,9+i)
        plt.imshow(mask[i])
    plt.show()


    epochs = 200
    n=0
    while True:
        im,mask,params = next(gen)
        loss,acc = model.train_on_batch(im,params)
        print "Iter:%i loss: %d, mae: %d"%(n,loss,acc)
        _lr = tf.to_float(model.optimizer.lr, name='ToFloat')
        sys.stdout.write("\033[F") # Cursor up one line

        n=n+1
        if n==epochs:
            break
    print "\nTrain finished! Eval"
    im,mask,params = next(InputGenerator(size=size,batch_size=1024))
    loss,mae =model.evaluate(im,params) 
    print "\nloss:%d, mae:%d"%(loss,mae)
    im,mask,params = next(InputGenerator(size=size,batch_size=16))
    print params
    print model.predict(im)

    

def create_model(inp_shape,output):
    i = Input(shape=inp_shape)
    x = _conv(i,96,sh=(3,3),strides=3)
    x = MaxPooling2D((2,2))(x)
    x = _conv(x,96)
    x = _conv(x,96)
    x = MaxPooling2D((2,2))(x)
    x = _conv(x,96)
    x = MaxPooling2D((2,2))(x)
    x= _conv(x,196,sh=(1,1))
    x = AveragePooling2D((2,2))(x)

    x = Flatten()(x)
    o = Dense(units=output, activation='relu')(x)
    model = Model(inputs = i,outputs = o,name = "Posnet")
    return model

def _conv(x,fls,sh=(3,3),strides=(1,1),bias =True,mom=0.9):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    x = Conv2D(fls,sh, strides= strides, use_bias=bias)(x)
    x = BatchNormalization(axis=channel_axis,momentum=mom)(x)
    x = Activation('relu')(x)
    return x
__main__()
