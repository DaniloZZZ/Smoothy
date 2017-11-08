from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape, Input
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout
from keras.models import Model, load_model

from keras.layers import Dropout, BatchNormalization, Reshape, Flatten, RepeatVector

from keras.layers import BatchNormalization
from keras.layers import Lambda, Dense, Input, Conv2D, MaxPool2D, UpSampling2D, concatenate
from keras.optimizers import Adam, RMSprop

from keras import backend as K
import tensorflow as tf

class GAModel(object):
    def __init__(self,num_classes,latent_dim, img_rows=28, img_cols=28, channel=1):

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channel = channel
        self.num_classes = num_classes
        self. latent_dim = latent_dim

        self.sess = tf.Session()
        K.set_session(self.sess)

        self.D = None   # discriminator
        self.G = None   # generator
        self.AM = None  # adversarial model
        self.DM = None  # discriminator model

    def discriminator_model(self):
        if self.DM:
            return self.DM
        optimizer = RMSprop(lr=0.0002, decay=6e-8)
        self.DM = Sequential()
        self.DM.add(self.discriminator())
        self.DM.compile(loss='mean_squared_error', optimizer=optimizer,\
            metrics=['mae'])
        return self.DM
    
    def get_tf_models(self):
        x_ = tf.placeholder(tf.float32, 
                shape=(None, self.img_cols,self.img_rows, self.channel),
                name='image')
        y_ = tf.placeholder(tf.float32, shape=(None, self.num_classes), name='labels')
        z_ = tf.placeholder(tf.float32, shape=(None, self.latent_dim),  name='z')

        img = Input(tensor=x_)
        self.img = img
        lbl = Input(tensor=y_)
        self.lbl = lbl
        z   = Input(tensor=z_)
        self.z  =z
        dropout_rate = 0.2

	with tf.variable_scope('generator'):
            thick = 96
	    x = concatenate([z, lbl])
	    x = Dense(8*8*thick, activation='relu')(x)
	    x = Dropout(dropout_rate)(x)
	    x = Reshape((8, 8, thick))(x)
	    x = UpSampling2D(size=(2, 2))(x)

	    x = Conv2D(96, kernel_size=(3, 3), activation='relu', padding='same')(x)
	    x = Dropout(dropout_rate)(x)

	    x = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(x)
	    x = Dropout(dropout_rate)(x)

	    x = UpSampling2D(size=(2, 2))(x)
	    x = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(x)

	    generated = Conv2D(self.channel, kernel_size=(5, 5), activation='sigmoid', padding='same')(x)
	generator = Model([z, lbl], generated, name='generator')


	with tf.variable_scope('discrim'):
	    x = Conv2D(96, kernel_size=(5, 5), strides=(2, 2), padding='same')(img)
	    x = LeakyReLU()(x)
	    x = Dropout(dropout_rate)(x)
	    x = MaxPool2D((2, 2), padding='same')(x)
	    
	    l = Conv2D(128, kernel_size=(3, 3), padding='same')(x)
	    l = Conv2D(128, kernel_size=(3, 3), padding='same')(x)
	    x = self.add_units_to_conv2d(l, lbl)
	    x = LeakyReLU()(l)
	    x = Dropout(dropout_rate)(x)

	    h = Flatten()(x)
	    d = Dense(1, activation='sigmoid')(h)
	discrim = Model([img, lbl], d, name='Discriminator')

	generated_z = generator([z, lbl])
        self.generated = generated_z
	discr_img   = discrim([img, lbl])
	discr_gen_z = discrim([generated_z, lbl])

	gan_model = Model([z, lbl], discr_gen_z, name='GAN')
        gan_model.summary()
	gan   = gan_model([z, lbl])

	log_dis_img   = tf.reduce_mean(-tf.log(discr_img + 1e-10))
	log_dis_gen_z = tf.reduce_mean(-tf.log(1. - discr_gen_z + 1e-10))

	self.L_gen = -log_dis_gen_z
	self.L_dis = 0.5*(log_dis_gen_z + log_dis_img)

	optimizer_gen = tf.train.RMSPropOptimizer(0.0001)
	optimizer_dis = tf.train.RMSPropOptimizer(0.0002)

	generator_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "generator")
	discrim_vars   = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "discrim")

	self.step_gen = optimizer_gen.minimize(self.L_gen, var_list=generator_vars)
	self.step_dis = optimizer_dis.minimize(self.L_dis, var_list=discrim_vars)

    def step(self,image, label, zp):
	l_dis, _ = self.sess.run([self.L_dis, self.step_gen], 
		feed_dict={self.z:zp, self.lbl:label, self.img:image, K.learning_phase():1})
	return l_dis

    def step_d(self,image, label, zp):
	l_dis, _ = self.sess.run([self.L_dis, self.step_dis],
		 feed_dict={self.z:zp, self.lbl:label, self.img:image, K.learning_phase():1})
	return l_dis

    def tf_gen(self, params, lat):
        return self.sess.run(self.generated, 
                feed_dict ={self.z:lat,self.lbl:params, K.learning_phase():0})

    def add_units_to_conv2d(self,conv2, units):
	dim1 = int(conv2.shape[1])
	dim2 = int(conv2.shape[2])
	dimc = int(units.shape[1])
	repeat_n = dim1*dim2
        print conv2.shape
	units_repeat = RepeatVector(repeat_n)(units)
	units_repeat = Reshape((dim1, dim2, dimc))(units_repeat)
	a = concatenate([conv2, units_repeat])
        return a


    def adversarial_model(self):
        if self.AM:
            return self.AM
        optimizer = RMSprop(lr=0.0001, decay=3e-8)
        self.AM = Sequential()
        self.AM.add(self.generator())
        self.AM.add(self.discriminator())
        self.AM.compile(loss='mean_squared_error', optimizer=optimizer,\
            metrics=['mae'])
        return self.AM

    def discriminator(self):
        if self.D:
            return self.D
        self.D = Sequential()
        depth = 64
        dropout = 0.3
        # In: 28 x 28 x 1, depth = 1
        # Out: 14 x 14 x 1, depth=64
        input_shape = (self.img_rows, self.img_cols, self.channel)
        self.D.add(Conv2D(depth*1, 5, strides=2, input_shape=input_shape,\
            padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))

        self.D.add(Conv2D(depth*2, 5, strides=2, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))

        self.D.add(Conv2D(depth*4, 5, strides=2, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*8, 3, strides=1, padding='same'))
        self.D.add(LeakyReLU(alpha=0.1))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*8, (1,3), strides=1, padding='same'))
        self.D.add(Conv2D(depth*8, (3,1), strides=1, padding='same'))
        self.D.add(LeakyReLU(alpha=0.1))
        self.D.add(Dropout(dropout))

        # Out: 1-dim probability
        self.D.add(Flatten())
        self.D.add(Dense(4))
        self.D.add(Activation('relu'))
        self.D.summary()
        return self.D

    def generator(self):
        if self.G:
            return self.G
        self.G = Sequential()
        dropout = 0.4
        depth = 64+64+64+64
        dim = 8
        # In: 100
        # Out: dim x dim x depth
        self.G.add(Dense(dim*dim*depth, input_dim=10))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))
        self.G.add(Reshape((dim, dim, depth)))

        # In: dim x dim x depth
        # Out: 2*dim x 2*dim x depth/2
        self.G.add(UpSampling2D())
        self.G.add(Conv2DTranspose(int(depth/2), 3, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        self.G.add(Conv2DTranspose(int(depth/2), 3, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        self.G.add(UpSampling2D())
        self.G.add(Conv2DTranspose(int(depth/4), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        self.G.add(Conv2DTranspose(int(depth/8), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        # Out: 28 x 28 x 3 image [0.0,1.0] per pix
        self.G.add(Conv2DTranspose(3, 5, padding='same'))
        self.G.add(Activation('sigmoid'))
        self.G.summary()
        return self.G
 
