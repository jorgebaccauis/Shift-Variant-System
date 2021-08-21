#!python3
import tensorflow as tf
import os
from matplotlib import pyplot as plt

from os import listdir
from os.path import isfile, join
from random import choices
import numpy as np




#-------------------- This are main py------------------------
from Read_Spectral import *   # data set build
from recoverynet_super import *     # Net build

def root_mean_squared_error(y_true, y_pred):
    return 20*tf.reduce_mean(tf.norm(y_true - y_pred, ord=2, axis=-1)) + tf.reduce_mean(tf.norm(y_true - y_pred, ord='fro', axis=[1,2]))


#----------------------------- directory of the spectral data set -------------------------
PATH = '/home/jams/Documents/dataset/Formated/'
PATH2 = '../dataset/Formated/'
# parameters of the net
BATCH_SIZE = 4; IMG_WIDTH = 250; IMG_HEIGHT = 250; L_bands    = 25; L_imput    = 25
test_dataset,train_dataset=Build_data_set(IMG_WIDTH=IMG_WIDTH,IMG_HEIGHT=IMG_HEIGHT,
                                         L_bands=L_bands,L_imput=L_imput,BATCH_SIZE=BATCH_SIZE,PATH=PATH)  # build the DataStore from Read_Spectral.py

diam = 3e-6
#-------------Net_model----------------------------------------------------------------
model = Proposed_net(input_size=(IMG_HEIGHT,IMG_WIDTH,L_imput),depth=25,depth_out=25,diam=diam)      # build the Net from recoverynet.py
print(model.summary())
from tensorflow.keras.callbacks import LearningRateScheduler


# This is a sample of a scheduler I used in the past
def lr_scheduler(epoch, lr):
    decay_step = 40
    if epoch % decay_step == 39 and epoch:
        lr = lr / 2
        tf.print(' Learning rate =' + str(lr))
        return lr

    return lr
optimizad = tf.keras.optimizers.Adam(learning_rate=1e-3)
def PSNR_Metric(y_true, y_pred):
  return tf.reduce_mean(tf.image.psnr(y_true,y_pred,1))
model.compile(optimizer=optimizad, loss=root_mean_squared_error, metrics = [PSNR_Metric],run_eagerly=False)
history = model.fit(train_dataset, epochs=150, validation_data=test_dataset,callbacks=[LearningRateScheduler(lr_scheduler, verbose=1)])
model.save_weights("modelo_free_3mm_"+str(diam)+".h5")


# See the height_map
temporal = model.get_weights()
polinomios=temporal[0]
zernike_volume = np.load('zernike_volume1_%d.npy' % 1000)
height_map = np.sum(polinomios * zernike_volume, axis=0)
plt.figure()
plt.imshow(height_map)
plt.colorbar()
plt.show()

## See some reconstruction
Img_spectral = loadmat('../dataset/Nueva carpeta/newIDS_COLORCHECK_1020-1215-1.mat')
Ref_img = Img_spectral['rad'][150:150+250,150:150+250,3:-3]
temp = Ref_img[:,:,[20,10,3]]/np.max( Ref_img[:,:,[20,10,3]])
plt.figure()
plt.imshow(temp),plt.title('Ideal')
plt.colorbar()
plt.show()
Ref_img=np.expand_dims(Ref_img,0)

Img_spectral = loadmat('../dataset/Nueva carpeta/newIDS_COLORCHECK_1020-1215-1.mat')
Ref_img = Img_spectral['rad'][150:150+250,150:150+250,3:-3]
Ref_img = Ref_img/np.max( Ref_img)
temp = Ref_img[:,:,[20,10,3]]
plt.figure()
plt.imshow(temp),plt.title('reference')
plt.colorbar()
plt.show()
Ref_img=np.expand_dims(Ref_img,0)
Resul= model.predict(Ref_img,batch_size=1)
Resul=Resul[0,:,:,:]
temp1 = Resul[:,:,[20,10,3]]/np.max(np.abs(Resul[:,:,[20,10,3]]))
plt.figure()
plt.imshow(temp1),plt.title('Recovered')
plt.colorbar()
plt.show()



Img_spectral = loadmat('../dataset/Nueva carpeta/new4cam_0411-1640-1.mat')
Ref_img = Img_spectral['rad'][150:150+250,150:150+250,3:-3]
Ref_img = Ref_img/np.max( Ref_img)
temp = Ref_img[:,:,[20,10,3]]
plt.figure()
plt.imshow(temp),plt.title('reference')
plt.colorbar()
plt.show()
Ref_img=np.expand_dims(Ref_img,0)
Resul= model.predict(Ref_img,batch_size=1)
Resul=Resul[0,:,:,:]
temp1 = Resul[:,:,[20,10,3]]/np.max(np.abs(Resul[:,:,[20,10,3]]))
plt.figure()
plt.imshow(temp1),plt.title('Recovered')
plt.colorbar()
plt.show()


