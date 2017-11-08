import numpy as np
import time
from tensorflow.examples.tutorials.mnist import input_data

from InpGen import InputGenerator
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from keras import backend as K
import tensorflow as tf

from GaModel import GAModel
class ElapsedTimer(object):
    def __init__(self):
        self.start_time = time.time()
    def elapsed(self,sec):
        if sec < 60:
            return str(sec) + " sec"
        elif sec < (60 * 60):
            return str(sec / 60) + " min"
        else:
            return str(sec / (60 * 60)) + " hr"
    def elapsed_time(self):
        print("Elapsed: %s " % self.elapsed(time.time() - self.start_time) )

class MNIST_DCGAN(object):
    def __init__(self):
        self.img_rows = 32
        self.img_cols = 32
        self.channel = 3
        self.latent_dim = 6
        self.num_classes = 4

        self.DCGAN = GAModel(
                latent_dim = self.latent_dim,
                num_classes = self.num_classes,
                img_cols= self.img_cols,
                img_rows=self.img_rows,
                channel = 3)

        self.DCGAN.get_tf_models()
        #self.discriminator =  self.DCGAN.discriminator_model()
        #self.adversarial = self.DCGAN.adversarial_model()
        #self.generator = self.DCGAN.generator()


    def train(self, train_steps=200, batch_size=256, save_interval=0):
        print "\niniting vars:"
	self.DCGAN.sess.run(tf.global_variables_initializer())

        gen = InputGenerator(ratio_range=(0.5,0.8),
                size=(self.img_cols,self.img_rows),
                batch_size = batch_size)

        if save_interval>0:
            params =np.array([[3*i,4,20,20] for i in range(1,5)])
            params= np.concatenate((params,[[4,3*i,20,20] for i in range(1,5)]))
            params= np.concatenate((params,[[4,4,3*i,20] for i in range(5,9)]))
            params= np.concatenate((params,[[4,4,20,3*i] for i in range(5,9)]))
            self.visparams =params
            print "params for vis:\n",params
            #genetate images for test params
            im = []
            for p in params:
                im.append(gen.gen_im(p)[0])
            self.visimages = np.array(im)
            self.plot_images(save2file=True,fake=False)

            self.latent= np.random.uniform(-1.0,1.0,size=[16,self.latent_dim])

        for i in range(train_steps):
            A_iter =5
            D_iter = 3
            lat = self.latent_dim

            images_train,masks,params = next(gen)
            inp = np.random.uniform(-1.0,1.0,size=[batch_size,lat])
            #inp = np.concatenate((inp,params),axis=1)

            for k in range(D_iter):
                images_train,masks,params = next(gen)
		l_d = self.DCGAN.step_d(images_train, params,inp)
                inp = np.random.uniform(-1.0,1.0,size=[batch_size,lat])
                #d_loss = self.discriminator.train_on_batch(images_train, params)

            for k in range(A_iter):
		l = self.DCGAN.step(images_train, params,inp)
                images_train,masks,params = next(gen)
                inp = np.random.uniform(-1.0,1.0,size=[batch_size,lat])
                #a_loss = self.adversarial.train_on_batch(inp, params)
            print "%i[D loss: %f , A loss: %f ]"%(i,l_d,l)
            #log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
            #log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
            #print(log_mesg)
            if save_interval>0:
                if (i+1)%save_interval==0:
                    self.plot_images(save2file=True,step=(i+1))

    def plot_images(self, save2file=False, fake=True, samples=16,  step=0):
        d = "img/params/"
        filename = d+'true.png'
        if fake:
            filename = d+"banana_%d.png" % step
            images = self.DCGAN.tf_gen(self.visparams,self.latent)
        else:
            images = self.visimages[:samples,:,:]

        plt.figure(figsize=(4,4))
        gs1 = gridspec.GridSpec(4, 4)
        for i in range(images.shape[0]):
            plt.subplot(gs1[i])
            image = images[i, :, :, :]
            plt.imshow(image, cmap='gray')
            plt.axis('off')
        gs1.update(wspace=0.05, hspace=0.07) # set the spacing between axes. 
        if save2file:
            plt.savefig(filename)
            plt.close('all')
        else:
            plt.show()

if __name__ == '__main__':
    mnist_dcgan = MNIST_DCGAN()
    timer = ElapsedTimer()
    mnist_dcgan.train(train_steps=20000, batch_size=128, save_interval=20)
    timer.elapsed_time()
    mnist_dcgan.plot_images(fake=True)
    mnist_dcgan.plot_images(fake=False, save2file=True)
