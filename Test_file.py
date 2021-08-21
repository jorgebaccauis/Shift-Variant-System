#!/home/jorgehdsp/.venv/tensorflow/bin/python
import sys

sys.path.append('/home/jorgehdsp/.venv/tensorflow/bin/python')
import tensorflow as tf
import os
from matplotlib import pyplot as plt

from os import listdir
from os.path import isfile, join
from random import choices
import numpy as np


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

#-------------------- This are main py------------------------
from Read_Spectral import *   # data set build
from recoverynet_super import *     # Net build

def root_mean_squared_error(y_true, y_pred):
    return 20*tf.reduce_mean(tf.norm(y_true - y_pred, ord=2, axis=-1)) + tf.reduce_mean(tf.norm(y_true - y_pred, ord='fro', axis=[1,2]))


#----------------------------- directory of the spectral data set -------------------------
#PATH = '/media/hdsp-deep/A2CC8AC9CC8A96E7/Spectral_data_set/data_500_spta_512_band_24/' # Carga de datos
PATH = '../dataset/Formated/'
PATH2 = '../dataset/Formated/'
# parameters of the net
BATCH_SIZE = 4; IMG_WIDTH = 250; IMG_HEIGHT = 250; L_bands    = 25; L_imput    = 25

#---------------------- Important Parameters to Compare --------------------------------------
diam = 3e-6
name_file = 'modelo_warp_3mm_5cm_1cmm_CA.h5'
clip = 0
wrap = 1 # see in the propagation change
File_name = 'R_len3_v2'
try:
    os.stat(File_name)
except:
    os.mkdir(File_name)

model = Proposed_net(input_size=(IMG_HEIGHT,IMG_WIDTH,L_imput),depth=L_imput,diam=diam)      # build the Net from recoverynet.py
model.load_weights(name_file)
model.compile(run_eagerly=True)
model_psfs  = Psf_show(input_size=(IMG_HEIGHT,IMG_WIDTH,L_imput),depth=L_imput,diam=diam)
it=1
model_psfs.layers[it].set_weights(model.layers[it].get_weights())

def PSNR_Metric(y_true, y_pred):
  return tf.reduce_mean(tf.image.psnr(y_true,y_pred,1))


# See the height_map
temporal = model_psfs.get_weights()
polinomios=temporal[0]
zernike_volume = np.load('zernike_volume1_%d.npy' % 1000)
height_map = np.sum(polinomios * zernike_volume, axis=0)
scipy.io.savemat(File_name + "/Heigh.mat",{'height_map': height_map})
plt.figure()
plt.imshow(height_map),plt.title('Height Map_before')
plt.colorbar()
plt.savefig(File_name+'/Height_before.png')
plt.show()

if(clip==1):
    height_map = tf.clip_by_value(height_map, -0.755e-6, 0.755e-6).numpy()
if(wrap==1):
    #height_map = ((((height_map / 0.755e-6) * np.pi + np.pi) % (2 * np.pi) - np.pi) / np.pi) * 0.755e-6
    height_map = tf.math.floormod(height_map + 0.755e-6, 2*0.755e-6 ) - 0.755e-6

height_map = height_map-np.min(height_map)
plt.figure()
plt.imshow(height_map),plt.title('Height Map')
plt.colorbar()
plt.savefig(File_name+'/Height.png')
plt.show()



## See some reconstruction
Img_spectral = loadmat('../dataset/Nueva carpeta/newIDS_COLORCHECK_1020-1215-1.mat')
Ref_img = Img_spectral['rad'][150:150+250,150:150+250,3:-3]
Ref_img = Ref_img/np.max( Ref_img)
plt.figure()
temp = Ref_img[:,:,[20,10,3]]/np.max( Ref_img[:,:,[20,10,3]])
scipy.io.savemat(File_name + "/Reference_1.mat",{'Ref_img': Ref_img})
plt.imshow(temp),plt.title('reference')
plt.savefig(File_name+'/reference_1.png')
plt.show()
Ref_img=np.expand_dims(Ref_img,0)
Temp = model_psfs.predict(Ref_img,batch_size=1)
Temp[0][0,:,:,:] = Temp[0][0,:,:,:]/np.max(Temp[0][0,:,:,:])
psfs_all = Temp[4]
plt.imshow(psfs_all[:,:,14])
plt.show()
plt.subplot(2,2,1),plt.imshow(Temp[0][0,:,:,:]),plt.title('Measurement')
plt.subplot(2,2,2),plt.imshow(Temp[1][0,:,:,0,0]),plt.title('psf 460')
plt.subplot(2,2,3),plt.imshow(Temp[2][0,:,:,0,0]),plt.title('psf 530')
plt.subplot(2,2,4),plt.imshow(Temp[3][0,:,:,0,0]),plt.title('psf 630')
plt.savefig(File_name+'/psfs_1.png'),plt.show()
Resul= model.predict(Ref_img,batch_size=1)
Resul=Resul[0,:,:,:]
scipy.io.savemat(File_name + "/psfs_all.mat",{'Temp': psfs_all})
scipy.io.savemat(File_name + "/psfs_460.mat",{'Temp': Temp[1][0,:,:,0,0]})
scipy.io.savemat(File_name + "/psfs_530.mat",{'Temp': Temp[2][0,:,:,0,0]})
scipy.io.savemat(File_name + "/psfs_630.mat",{'Temp': Temp[3][0,:,:,0,0]})
scipy.io.savemat(File_name + "/Resul_1.mat",{'Resul': Resul})
temp1 = Resul[:,:,[20,10,3]]/np.max(np.abs(Resul[:,:,[20,10,3]]))
plt.figure()
plt.imshow(temp1),plt.title('Recovered ='+str(PSNR_Metric(Ref_img,Resul).numpy()))
plt.savefig(File_name+'/Recovered_1.png')
plt.show()

## See some reconstruction
Img_spectral = loadmat('../dataset/Nueva carpeta/newBGU_0403-1419-1.mat')
Ref_img = Img_spectral['rad'][150:150+250,150:150+250,3:-3]
Ref_img = Ref_img/np.max(Ref_img)
plt.figure()
temp = Ref_img[:,:,[20,10,3]]/np.max( Ref_img[:,:,[20,10,3]])
scipy.io.savemat(File_name + "/Reference_2.mat",{'Ref_img': Ref_img})
plt.imshow(temp),plt.title('reference')
plt.savefig(File_name+'/reference_2.png')
plt.show()
Ref_img=np.expand_dims(Ref_img,0)
Temp = model_psfs.predict(Ref_img,batch_size=1)
Temp[0][0,:,:,:] = Temp[0][0,:,:,:]/np.max(Temp[0][0,:,:,:])
plt.subplot(2,2,1),plt.imshow(Temp[0][0,:,:,:]),plt.title('Measurement')
plt.subplot(2,2,2),plt.imshow(Temp[1][0,:,:,0,0]),plt.title('psf 460')
plt.subplot(2,2,3),plt.imshow(Temp[2][0,:,:,0,0]),plt.title('psf 530')
plt.subplot(2,2,4),plt.imshow(Temp[3][0,:,:,0,0]),plt.title('psf 630')
plt.savefig(File_name+'/psfs_2.png'),plt.show()
Resul= model.predict(Ref_img,batch_size=1)
Resul=Resul[0,:,:,:]
scipy.io.savemat(File_name + "/Resul_2.mat",{'Resul': Resul})
temp1 = Resul[:,:,[20,10,3]]/np.max(np.abs(Resul[:,:,[20,10,3]]))
plt.figure()
plt.imshow(temp1),plt\
    .title('Recovered ='+str(PSNR_Metric(Ref_img,Resul).numpy()))
plt.savefig(File_name+'/Recovered_2.png')
plt.show()

## See some reconstruction
Img_spectral = loadmat('../dataset/Nueva carpeta/new4cam_0411-1640-1.mat')
Ref_img = Img_spectral['rad'][150:150+250,150:150+250,3:-3]
Ref_img = Ref_img/np.max( Ref_img)
plt.figure()
temp = Ref_img[:,:,[20,10,3]]/np.max( Ref_img[:,:,[20,10,3]])
scipy.io.savemat(File_name + "/Reference_3.mat",{'Ref_img': Ref_img})
plt.imshow(temp),plt.title('reference')
plt.savefig(File_name+'/reference_3.png')
plt.show()
Ref_img=np.expand_dims(Ref_img,0)
Temp = model_psfs.predict(Ref_img,batch_size=1)
Temp[0][0,:,:,:] = Temp[0][0,:,:,:]/np.max(Temp[0][0,:,:,:])
plt.subplot(2,2,1),plt.imshow(Temp[0][0,:,:,:]),plt.title('Measurement')
plt.subplot(2,2,2),plt.imshow(Temp[1][0,:,:,0,0]),plt.title('psf 460')
plt.subplot(2,2,3),plt.imshow(Temp[2][0,:,:,0,0]),plt.title('psf 530')
plt.subplot(2,2,4),plt.imshow(Temp[3][0,:,:,0,0]),plt.title('psf 630')
plt.savefig(File_name+'/psfs_3.png'),plt.show()
Resul= model.predict(Ref_img,batch_size=1)
Resul=Resul[0,:,:,:]
scipy.io.savemat(File_name + "/Resul_3.mat",{'Resul': Resul})
temp1 = Resul[:,:,[20,10,3]]/np.max(np.abs(Resul[:,:,[20,10,3]]))
plt.figure()
plt.imshow(temp1),plt.title('Recovered ='+str(PSNR_Metric(Ref_img,Resul).numpy()))
plt.savefig(File_name+'/Recovered_3.png')
plt.show()