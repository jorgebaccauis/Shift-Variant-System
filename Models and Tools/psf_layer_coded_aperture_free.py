#!python3
import tensorflow as K  # se puede cambiar por from keras.import backend as K
from tensorflow.keras.layers import Layer  # quitar tensorflow si usa keras solo
from tensorflow.keras.constraints import NonNeg
import numpy as np
import poppy
import os
from random import random
from scipy.io import loadmat
from functions import reg_binary, deta, area_downsampling_tf, img_psf_conv,get_color_bases,propagation,propagation_back,kronecker_product
from tensorflow.keras.constraints import NonNeg

class Psf_layer_mostrar(Layer):


    def __init__(self, output_dim, distance=50e-3, patch_size=250, sample_interval=3.69e-6, wave_resolution=500,
                 wave_lengths=None, distance_code = 47e-3,Nt=5,fac_m=1,refractive_idcs=None, nterms=230, bgr_response=None, height_tolerance=20e-9,
                 **kwargs):
        self.output_dim = output_dim
        self.distance = distance  # critico
        self.height_tolerance = height_tolerance # manufacturing error
        self.fac_m = fac_m
        if bgr_response is not None:
            self.bgr_response = K.cast(bgr_response,dtype=K.float32)
        else:
            temp = loadmat('Sensor_12.mat')
            self.bgr_response = np.concatenate((temp['B'], temp['G'], temp['R']))
            #self.bgr_response = K.cast(self.bgr_response, dtype=K.float32)
            #self.bgr_response = K.expand_dims(K.expand_dims(K.expand_dims(self.bgr_response, -1), -1), -1)

        if wave_lengths is not None:
            self.wave_lengths = K.cast(wave_lengths,dtype=K.float32)
        else:
            self.wave_lengths = K.cast(np.linspace(420, 660, 25)*1e-9,dtype=K.float32)
        if refractive_idcs is not None:
            self.refractive_idcs = K.cast(refractive_idcs,dtype=K.float32)
        else:
            self.refractive_idcs = deta(self.wave_lengths)
        self.patch_size = patch_size  # Size of patches to be extracted from images, and resolution of simulated sensor
        self.sample_interval = sample_interval  # Sampling interval (size of one "pixel" in the simulated wavefront)
        self.wave_resolution = wave_resolution,wave_resolution

        self.distance_code = distance_code
        self.Nt = Nt

        self.fr, self.fg, self.fc, self.fb = get_color_bases(self.wave_lengths)
        fr = K.transpose(self.fr, [2, 0, 1])
        fg = K.transpose(self.fg, [2, 0, 1])
        fc = K.transpose(self.fc, [2, 0, 1])
        fb = K.transpose(self.fb, [2, 0, 1])
        self.ft = K.concat([fr, fg], axis=1)
        self.ft = K.concat([self.ft, fc], axis=1)
        self.ft = K.concat([self.ft, fb], axis=1)
        self.ft = K.expand_dims(self.ft, axis=-1)

        if not os.path.exists('zernike_volume1_%d.npy' % self.wave_resolution[0]):
            self.zernike_volume = (
                        1e-6 * poppy.zernike.zernike_basis(nterms=nterms, npix=self.wave_resolution[0], outside=0.0)).astype(
                np.float32)
            np.save('zernike_volume1_%d.npy' % self.wave_resolution[0], self.zernike_volume)
        else:
            self.zernike_volume = np.load('zernike_volume1_%d.npy' % self.wave_resolution[0])
        super(Psf_layer_mostrar, self).__init__(**kwargs)

    def build(self, input_shape):
        num_zernike_coeffs = self.zernike_volume.shape[0]
        zernike_inits = np.zeros((num_zernike_coeffs, 1, 1))
        #zernike_inits[3] = -0.5  # This sets the defocus value to approximately focus the image for a distance of 1m.
        zernike_initializer = K.constant_initializer(zernike_inits)
        self.zernike_coeffs = self.add_weight(name='zernike_coeffs', shape=(zernike_inits.shape),
                                              initializer=zernike_initializer, trainable=True)

        self.Mask_pattern = self.add_weight(name='mask', shape=(self.Nt, self.Nt,25),
                                            initializer=K.random_normal_initializer, trainable=True,
                                            regularizer=reg_binary)
        #self.wr = K.expand_dims(self.wr, -1)
        #self.wg = K.expand_dims(self.wg, -1)
        #self.wb = K.expand_dims(self.wb, -1)
        #[self.x, self.y] = np.mgrid[-input_shape[1] : input_shape[1] ,
        #         -input_shape[2] : input_shape[2] ].astype(np.float32)
        [self.x, self.y] = np.mgrid[-self.wave_resolution[0]//2: self.wave_resolution[0]//2,
                                    -self.wave_resolution[0]//2 : self.wave_resolution[0]//2 ].astype(np.float32)
        self.max_val = np.amax(self.x)
        # max_val = int(self.patch_len / 2)
        self.myweights = K.convert_to_tensor(loadmat('FiltroEd.mat')['Fp2']/2, dtype=K.float32)
        self.r = np.sqrt(self.x ** 2 + self.y ** 2)[None, :, :, None]
        self.aperture = (self.r < self.max_val).astype(np.float32)
        self.x = K.convert_to_tensor(self.x, dtype=K.float32)
        self.y = K.convert_to_tensor(self.y, dtype=K.float32)
        self.max_val = K.convert_to_tensor(self.max_val, dtype=K.float32)
        self.r = K.convert_to_tensor(self.r, dtype=K.float32)
        self.aperture = K.convert_to_tensor(self.aperture,dtype=K.complex64)
        N_psf = K.floor((self.Nt * 2 * self.fac_m) * self.patch_size / self.wave_resolution[0])
        self.St = None
        fac = int(K.floor(self.patch_size / N_psf))
        for rec in range(int(N_psf)):
            for rec2 in range(int(N_psf)):
                aux = np.zeros((int(N_psf), int(N_psf)))
                aux[rec, rec2] = 1
                S = np.kron(np.ones((fac, fac)), aux)
                S = K.expand_dims(K.convert_to_tensor(S, dtype=K.float32), 0)
                if self.St is None:
                    self.St = K.expand_dims(S,-1)
                else:
                    self.St = K.concat([self.St,K.expand_dims(S,axis=-1)],-1)

        super(Psf_layer_mostrar, self).build(input_shape)

    def call(self, inputs, **kwargs):

        #Aux1 = K.sigmoid(self.Mask_pattern)
        #Aux1 = K.transpose(K.reduce_sum(self.ft * self.Mask_pattern, axis=1), [1, 2, 0])
        Aux1 = K.sigmoid(self.Mask_pattern)
        Aux1 = K.concat([Aux1, K.image.flip_left_right(Aux1)], axis=1)
        Aux1 = K.concat([Aux1, K.image.flip_up_down(Aux1)], axis=0)

        Mask = kronecker_product(
            K.ones((int(self.wave_resolution[0] / (2 * self.fac_m * self.Nt)),
                    int(self.wave_resolution[0] / (2 * self.fac_m * self.Nt)))), Aux1)

        height_map = self.myweights
        #height_map = K.reduce_sum(self.zernike_coeffs * self.zernike_volume, axis=0)
        height_map = K.expand_dims(K.expand_dims(height_map, 0), -1, name='height_map')
        #height_map = K.clip_by_value(height_map,-0.755e-6,0.755e-6)
        height_map = K.math.floormod(height_map + 0.755e-6, 2 * 0.755e-6) - 0.755e-6

        refractive_idcs1 = self.refractive_idcs
        wave_lengths1 = self.wave_lengths
        #psftotal = None
        N_psf = K.floor(
            (self.Nt * 2 * self.fac_m) * self.patch_size / self.wave_resolution[0])  # N_psf^2 number of psfs
        y_med_r  = K.TensorArray(K.float32, size=int(N_psf*N_psf*len(refractive_idcs1)))
        y_med_g = K.TensorArray(K.float32, size=int(N_psf*N_psf*len(refractive_idcs1)))
        y_med_b = K.TensorArray(K.float32, size=int(N_psf*N_psf*len(refractive_idcs1)))
        it = 0
        psfs_ban_3 = K.TensorArray(K.float32, size=1)
        psfs_ban_10 = K.TensorArray(K.float32, size=1)
        psfs_ban_20 = K.TensorArray(K.float32, size=1)
        for band in range(len(refractive_idcs1)):
            refractive_idcs = refractive_idcs1[band]
            wave_lengths = wave_lengths1[band]
            delta_N = K.reshape(refractive_idcs,[1, 1, 1, -1])


        # wave number
            wave_nos = 2. * np.pi / wave_lengths
            wave_nos = K.reshape(wave_nos,[1, 1, 1, -1])
        # ------------------------------------
        # phase delay indiced by height field
        # ------------------------------------

        # phi = height_map
            phase = wave_nos * delta_N * height_map
            field = K.add(K.cast(K.cos(phase), dtype=K.complex64),
                                1.j * K.cast(K.sin(phase), dtype=K.complex64),
                                name='phase_plate_shift')


            output_field_circ = self.aperture * field


            frac_N_Nl = self.patch_size / self.wave_resolution[0]  # decimation factor
            pix_shift = K.floor((-K.floor(self.patch_size / N_psf / 2) + self.patch_size / N_psf / 2) * 2 * N_psf)
            #K.print(N_psf)
            for rec in range(int(N_psf)):
                for rec2 in range(int(N_psf)):
                    movi = K.cast(K.floor(pix_shift - K.cast(rec,dtype=K.float32) * frac_N_Nl),dtype=K.int32)
                    movj = K.cast(K.floor(pix_shift - K.cast(rec2,dtype=K.float32) * frac_N_Nl), dtype=K.int32)
                    maskShif = K.roll(K.roll(Mask[:, :, band], movj, axis=1), movi, axis=0)
                    maskShif = K.expand_dims(K.expand_dims(K.cast(maskShif, dtype=K.complex64), 0), -1)
                    out_field1 = propagation(output_field_circ, self.sample_interval, wave_lengths, self.distance_code)
                    out_field1 = K.multiply(out_field1, maskShif)
                    out_field1 = propagation_back(out_field1, self.sample_interval, wave_lengths, self.distance_code)
                    out_field1 = propagation(out_field1, self.sample_interval, wave_lengths, self.distance)
                    #out_field1 = propagation(output_field_circ, self.sample_interval, wave_lengths, self.distance)

                    psfs = K.square(K.abs(out_field1), name='intensities')
                    # Downsample psf to image resolution & normalize to sum to 1
                    psfs = K.pad(psfs,[[0, 0], [3000, 3000], [3000, 3000], [0, 0]])
                    psfs = area_downsampling_tf(psfs, self.patch_size)
                    psfs = K.math.divide(psfs, K.reduce_sum(psfs, axis=[1, 2]))
                    psfs = K.transpose(psfs, [1, 2, 0, 3])
                    if(band==3):
                        psfs_ban_3 = psfs_ban_3.write(0,psfs)
                    if (band == 10):
                        psfs_ban_10 = psfs_ban_10.write(0,psfs)
                    if (band == 20):
                        psfs_ban_20 = psfs_ban_20.write(0,psfs)
                    output_image = img_psf_conv(K.expand_dims(inputs[:, :, :, band] * self.St[:,:,:,int(rec*int(N_psf)+rec2)], -1), psfs)

                    y_med_r=y_med_r.write(it,self.fr[0, 0, band] * output_image)
                    y_med_g=y_med_g.write(it,self.fg[0, 0, band] * output_image)
                    y_med_b=y_med_b.write(it,self.fb[0, 0, band] * output_image)
                    it = it + 1
        y_med_r = K.transpose(y_med_r.stack(),[1,2,3,4,0])
        y_med_r = K.reshape(y_med_r,[K.shape(y_med_r)[0],K.shape(y_med_r)[1],K.shape(y_med_r)[2],-1])
        y_med_g = K.transpose(y_med_g.stack(), [1, 2, 3, 4, 0])
        y_med_g = K.reshape(y_med_g, [K.shape(y_med_r)[0], K.shape(y_med_g)[1], K.shape(y_med_g)[2], -1])
        y_med_b = K.transpose(y_med_b.stack(), [1, 2, 3, 4, 0])
        y_med_b = K.reshape(y_med_b, [K.shape(y_med_b)[0], K.shape(y_med_b)[1], K.shape(y_med_b)[2], -1])
        y_med_r = K.reduce_sum(y_med_r, axis=3)
        y_med_r = K.expand_dims(y_med_r, -1)
        y_med_g = K.reduce_sum(y_med_g, axis=3)
        y_med_g = K.expand_dims(y_med_g, -1)
        y_med_b = K.reduce_sum(y_med_b, axis=3)
        y_med_b = K.expand_dims(y_med_b, -1)

        y_final = K.concat([y_med_r, y_med_g, y_med_b], axis=3)

        #y_final = K.divide(y_final, K.math.reduce_max(y_final))
        #y_final = K.keras.activations.sigmoid(y_final)

        return y_final,psfs_ban_3.stack(),psfs_ban_10.stack(),psfs_ban_20.stack()

class Psf_layer(Layer):


    def __init__(self, output_dim, distance=50e-3, patch_size=250, sample_interval=3.69e-6, wave_resolution=500,
                 wave_lengths=None, distance_code = 47e-3,Nt=5,fac_m=1,refractive_idcs=None, nterms=230, bgr_response=None, height_tolerance=20e-9,
                 **kwargs):
        self.output_dim = output_dim
        self.distance = distance  # critico
        self.height_tolerance = height_tolerance # manufacturing error
        self.fac_m = fac_m
        if bgr_response is not None:
            self.bgr_response = K.cast(bgr_response,dtype=K.float32)
        else:
            temp = loadmat('Sensor_12.mat')
            self.bgr_response = np.concatenate((temp['B'], temp['G'], temp['R']))
            #self.bgr_response = K.cast(self.bgr_response, dtype=K.float32)
            #self.bgr_response = K.expand_dims(K.expand_dims(K.expand_dims(self.bgr_response, -1), -1), -1)

        if wave_lengths is not None:
            self.wave_lengths = K.cast(wave_lengths,dtype=K.float32)
        else:
            self.wave_lengths = K.cast(np.linspace(420, 660, 25)*1e-9,dtype=K.float32)
        if refractive_idcs is not None:
            self.refractive_idcs = K.cast(refractive_idcs,dtype=K.float32)
        else:
            self.refractive_idcs = deta(self.wave_lengths)
        self.patch_size = patch_size  # Size of patches to be extracted from images, and resolution of simulated sensor
        self.sample_interval = sample_interval  # Sampling interval (size of one "pixel" in the simulated wavefront)
        self.wave_resolution = wave_resolution,wave_resolution

        self.distance_code = distance_code
        self.Nt = Nt

        self.fr, self.fg, self.fc, self.fb = get_color_bases(self.wave_lengths)
        fr = K.transpose(self.fr, [2, 0, 1])
        fg = K.transpose(self.fg, [2, 0, 1])
        fc = K.transpose(self.fc, [2, 0, 1])
        fb = K.transpose(self.fb, [2, 0, 1])
        self.ft = K.concat([fr, fg], axis=1)
        self.ft = K.concat([self.ft, fc], axis=1)
        self.ft = K.concat([self.ft, fb], axis=1)
        self.ft = K.expand_dims(self.ft, axis=-1)
        if not os.path.exists('zernike_volume1_%d.npy' % self.wave_resolution[0]):
            self.zernike_volume = (
                        1e-6 * poppy.zernike.zernike_basis(nterms=nterms, npix=self.wave_resolution[0], outside=0.0)).astype(
                np.float32)
            np.save('zernike_volume1_%d.npy' % self.wave_resolution[0], self.zernike_volume)
        else:
            self.zernike_volume = np.load('zernike_volume1_%d.npy' % self.wave_resolution[0])
        super(Psf_layer, self).__init__(**kwargs)

    def build(self, input_shape):
        num_zernike_coeffs = self.zernike_volume.shape[0]
        zernike_inits = np.zeros((num_zernike_coeffs, 1, 1))
        #zernike_inits[3] = -0.5  # This sets the defocus value to approximately focus the image for a distance of 1m.
        zernike_initializer = K.constant_initializer(zernike_inits)
        self.zernike_coeffs = self.add_weight(name='zernike_coeffs', shape=(zernike_inits.shape),
                                              initializer=zernike_initializer, trainable=True)

        self.Mask_pattern = self.add_weight(name='mask', shape=(4, self.Nt, self.Nt),
                                            initializer=K.random_normal_initializer, trainable=True,
                                            regularizer=reg_binary)
        #self.wr = K.expand_dims(self.wr, -1)
        #self.wg = K.expand_dims(self.wg, -1)
        #self.wb = K.expand_dims(self.wb, -1)
        #[self.x, self.y] = np.mgrid[-input_shape[1] : input_shape[1] ,
        #         -input_shape[2] : input_shape[2] ].astype(np.float32)
        [self.x, self.y] = np.mgrid[-self.wave_resolution[0]//2: self.wave_resolution[0]//2,
                                    -self.wave_resolution[0]//2 : self.wave_resolution[0]//2 ].astype(np.float32)
        self.max_val = np.amax(self.x)
        # max_val = int(self.patch_len / 2)

        self.r = np.sqrt(self.x ** 2 + self.y ** 2)[None, :, :, None]
        self.aperture = (self.r < self.max_val).astype(np.float32)
        self.x = K.convert_to_tensor(self.x, dtype=K.float32)
        self.y = K.convert_to_tensor(self.y, dtype=K.float32)
        self.max_val = K.convert_to_tensor(self.max_val, dtype=K.float32)
        self.r = K.convert_to_tensor(self.r, dtype=K.float32)
        self.aperture = K.convert_to_tensor(self.aperture,dtype=K.complex64)
        N_psf = K.floor((self.Nt * 2 * self.fac_m) * self.patch_size / self.wave_resolution[0])
        self.St = None
        fac = int(K.floor(self.patch_size / N_psf))
        for rec in range(int(N_psf)):
            for rec2 in range(int(N_psf)):
                aux = np.zeros((int(N_psf), int(N_psf)))
                aux[rec, rec2] = 1
                S = np.kron(np.ones((fac, fac)), aux)
                S = K.expand_dims(K.convert_to_tensor(S, dtype=K.float32), 0)
                if self.St is None:
                    self.St = K.expand_dims(S,-1)
                else:
                    self.St = K.concat([self.St,K.expand_dims(S,axis=-1)],-1)

        super(Psf_layer, self).build(input_shape)

    def call(self, inputs, **kwargs):

        #Aux1 = K.sigmoid(self.Mask_pattern)
        Aux1 = K.transpose(K.reduce_sum(self.ft * self.Mask_pattern, axis=1), [1, 2, 0])
        Aux1 = K.concat([Aux1, K.image.flip_left_right(Aux1)], axis=1)
        Aux1 = K.concat([Aux1, K.image.flip_up_down(Aux1)], axis=0)

        Mask = kronecker_product(
            K.ones((int(self.wave_resolution[0] / (2 * self.fac_m * self.Nt)),
                    int(self.wave_resolution[0] / (2 * self.fac_m * self.Nt)))), Aux1)


        height_map = K.reduce_sum(self.zernike_coeffs * self.zernike_volume, axis=0)
        height_map = K.expand_dims(K.expand_dims(height_map, 0), -1, name='height_map')
        #height_map = K.clip_by_value(height_map,-0.755e-6,0.755e-6)
        ##height_map = K.pad(height_map,
          ##                  [[0, 0], [self.Pad, self.Pad], [self.Pad, self.Pad], [0, 0]])


        #if self.height_tolerance is not None:
        #    height_map += K.random.uniform(shape=height_map.shape, minval=-self.height_tolerance,
        #                                   maxval=self.height_tolerance, dtype=height_map.dtype)
        #y * (0.5 - atan(cos(pi * x / y) / sin(pi * x / y)) / pi)

        #y = (2 * 3.1416)
        #x = ((height_map / 0.755e-6) * 3.1416 + 3.1416)
        #mod = y*(0.5-K.math.atan(K.math.cos(3.1416*x/y)/K.math.sin(3.1416*x/y))/3.1416)

        #height_map = ( (mod-3.1416)/ 3.1416) * 0.755e-6
        #height_map = ((((height_map / 0.755e-6) * np.pi + np.pi) % (2 * np.pi) - np.pi) / np.pi) * 0.755e-6
        #height_map = K.math.floormod(height_map + 0.755e-6, 0.755e-6) - 0.755e-6
        #height_map = K.math.floormod(height_map + 0.755e-6, 2 * 0.755e-6) - 0.755e-6

        refractive_idcs1 = self.refractive_idcs
        wave_lengths1 = self.wave_lengths
        #psftotal = None
        N_psf = K.floor(
            (self.Nt * 2 * self.fac_m) * self.patch_size / self.wave_resolution[0])  # N_psf^2 number of psfs
        y_med_r  = K.TensorArray(K.float32, size=int(N_psf*N_psf*len(refractive_idcs1)))
        y_med_g = K.TensorArray(K.float32, size=int(N_psf*N_psf*len(refractive_idcs1)))
        y_med_b = K.TensorArray(K.float32, size=int(N_psf*N_psf*len(refractive_idcs1)))
        it = 0
        for band in range(len(refractive_idcs1)):
            refractive_idcs = refractive_idcs1[band]
            wave_lengths = wave_lengths1[band]
            delta_N = K.reshape(refractive_idcs,[1, 1, 1, -1])


        # wave number
            wave_nos = 2. * np.pi / wave_lengths
            wave_nos = K.reshape(wave_nos,[1, 1, 1, -1])
        # ------------------------------------
        # phase delay indiced by height field
        # ------------------------------------

        # phi = height_map
            phase = wave_nos * delta_N * height_map
            field = K.add(K.cast(K.cos(phase), dtype=K.complex64),
                                1.j * K.cast(K.sin(phase), dtype=K.complex64),
                                name='phase_plate_shift')


            output_field_circ = self.aperture * field


            frac_N_Nl = self.patch_size / self.wave_resolution[0]  # decimation factor
            pix_shift = K.floor((-K.floor(self.patch_size / N_psf / 2) + self.patch_size / N_psf / 2) * 2 * N_psf)
            #K.print(N_psf)
            for rec in range(int(N_psf)):
                for rec2 in range(int(N_psf)):
                    movi = K.cast(K.floor(pix_shift - K.cast(rec,dtype=K.float32) * frac_N_Nl),dtype=K.int32)
                    movj = K.cast(K.floor(pix_shift - K.cast(rec2,dtype=K.float32) * frac_N_Nl), dtype=K.int32)
                    maskShif = K.roll(K.roll(Mask[:, :, band], movj, axis=1), movi, axis=0)
                    maskShif = K.expand_dims(K.expand_dims(K.cast(maskShif, dtype=K.complex64), 0), -1)
                    out_field1 = propagation(output_field_circ, self.sample_interval, wave_lengths, self.distance_code)
                    out_field1 = K.multiply(out_field1, maskShif)
                    out_field1 = propagation_back(out_field1, self.sample_interval, wave_lengths, self.distance_code)
                    out_field1 = propagation(out_field1, self.sample_interval, wave_lengths, self.distance)
                    #out_field1 = propagation(output_field_circ, self.sample_interval, wave_lengths, self.distance)

                    psfs = K.square(K.abs(out_field1), name='intensities')
                    # Downsample psf to image resolution & normalize to sum to 1
                    psfs = K.pad(psfs,[[0, 0], [3000, 3000], [3000, 3000], [0, 0]])
                    psfs = area_downsampling_tf(psfs, self.patch_size)
                    psfs = K.math.divide(psfs, K.reduce_sum(psfs, axis=[1, 2]))
                    psfs = K.transpose(psfs, [1, 2, 0, 3])
                    output_image = img_psf_conv(K.expand_dims(inputs[:, :, :, band] * self.St[:,:,:,int(rec*int(N_psf)+rec2)], -1), psfs)

                    y_med_r=y_med_r.write(it,self.fr[0, 0, band] * output_image)
                    y_med_g=y_med_g.write(it,self.fg[0, 0, band] * output_image)
                    y_med_b=y_med_b.write(it,self.fb[0, 0, band] * output_image)
                    it = it + 1
        y_med_r = K.transpose(y_med_r.stack(),[1,2,3,4,0])
        y_med_r = K.reshape(y_med_r,[K.shape(y_med_r)[0],K.shape(y_med_r)[1],K.shape(y_med_r)[2],-1])
        y_med_g = K.transpose(y_med_g.stack(), [1, 2, 3, 4, 0])
        y_med_g = K.reshape(y_med_g, [K.shape(y_med_r)[0], K.shape(y_med_g)[1], K.shape(y_med_g)[2], -1])
        y_med_b = K.transpose(y_med_b.stack(), [1, 2, 3, 4, 0])
        y_med_b = K.reshape(y_med_b, [K.shape(y_med_b)[0], K.shape(y_med_b)[1], K.shape(y_med_b)[2], -1])
        y_med_r = K.reduce_sum(y_med_r, axis=3)
        y_med_r = K.expand_dims(y_med_r, -1)
        y_med_g = K.reduce_sum(y_med_g, axis=3)
        y_med_g = K.expand_dims(y_med_g, -1)
        y_med_b = K.reduce_sum(y_med_b, axis=3)
        y_med_b = K.expand_dims(y_med_b, -1)

        y_final = K.concat([y_med_r, y_med_g, y_med_b], axis=3)

        #y_final = K.divide(y_final, K.math.reduce_max(y_final))
        #y_final = K.keras.activations.sigmoid(y_final)

        return y_final

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)



