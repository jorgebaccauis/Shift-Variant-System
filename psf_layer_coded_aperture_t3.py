import tensorflow as K  # se puede cambiar por from keras.import backend as K
from tensorflow.keras.layers import Layer  # quitar tensorflow si usa keras solo
from tensorflow.keras.constraints import NonNeg
import numpy as np
import poppy
import os
from random import random
from scipy.io import loadmat
from functions import reg_binary, deta, ifftshift, area_downsampling_tf, compl_exp_tf, transp_fft2d, transp_ifft2d, img_psf_conv,fftshift2d_tf,get_color_bases,propagation,propagation_back,kronecker_product
from tensorflow.keras.constraints import NonNeg


class Psf_layer(Layer):

    def __init__(self, output_dim, distance=50e-3, patch_size=250, sample_interval=3.69e-6, wave_resolution=500,
                 wave_lengths=None, distance_code = 47e-3,Nt=5,fac_m=1,refractive_idcs=None, nterms=230, bgr_response=None, height_tolerance=20e-9,
                 **kwargs):
        self.output_dim = output_dim
        self.distance = distance  # critico
        self.height_tolerance = height_tolerance # manufacturing error
        self.fac_m = fac_m
        if bgr_response is not None:
            self.bgr_response = bgr_response
        else:
            temp = loadmat('Sensor_12.mat')
            self.bgr_response = np.concatenate((temp['B'], temp['G'], temp['R']))
            self.bgr_response = K.cast(self.bgr_response, dtype=K.float32)
            self.bgr_response = K.expand_dims(K.expand_dims(K.expand_dims(self.bgr_response, -1), -1), -1)

        if wave_lengths is not None:
            self.wave_lengths = wave_lengths
        else:
            self.wave_lengths = np.linspace(420, 660, 24)*1e-9
        if refractive_idcs is not None:
            self.refractive_idcs = refractive_idcs
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
        zernike_inits[3] = -5  # This sets the defocus value to approximately focus the image for a distance of 1m.
        zernike_initializer = K.constant_initializer(zernike_inits)
        self.zernike_coeffs = self.add_weight(name='zernike_coeffs', shape=(zernike_inits.shape),
                                              initializer=zernike_initializer, trainable=True)


        self.Mask_pattern = self.add_weight(name='mask', shape=(4, self.Nt,self.Nt),
                                            initializer=K.random_normal_initializer, trainable=True,
                                            regularizer=reg_binary)
        #self.wr = K.expand_dims(self.wr, -1)
        #self.wg = K.expand_dims(self.wg, -1)
        #self.wb = K.expand_dims(self.wb, -1)

        super(Psf_layer, self).build(input_shape)

    def call(self, inputs, **kwargs):

        Aux1 = K.transpose(K.reduce_sum(self.ft * self.Mask_pattern, axis=1), [1, 2, 0])
        Aux1 = K.concat([Aux1, K.image.flip_left_right(Aux1)], axis=1)
        Aux1 = K.concat([Aux1, K.image.flip_up_down(Aux1)], axis=0)

        Mask = kronecker_product(
            K.ones((int(self.wave_resolution[0] / (2 * self.fac_m * self.Nt)), int(self.wave_resolution[0] / (2 * self.fac_m * self.Nt)))), Aux1)


        height_map = K.reduce_sum(self.zernike_coeffs * self.zernike_volume, axis=0)
        height_map = K.expand_dims(K.expand_dims(height_map, 0), -1, name='height_map')

        ##height_map = K.pad(height_map,
          ##                  [[0, 0], [self.Pad, self.Pad], [self.Pad, self.Pad], [0, 0]])


        if self.height_tolerance is not None:
            height_map += K.random.uniform(shape=height_map.shape, minval=-self.height_tolerance,
                                           maxval=self.height_tolerance, dtype=height_map.dtype)

        refractive_idcs1 = self.refractive_idcs.copy()
        wave_lengths1 = self.wave_lengths.copy()
        psftotal = None
        y_med_r = None
        y_med_g = None
        y_med_b = None
        for band in range(len(refractive_idcs1)):
            refractive_idcs = refractive_idcs1[band]
            wave_lengths = wave_lengths1[band]
            delta_N = refractive_idcs.reshape([1, 1, 1, -1])


        # wave number
            wave_nos = 2. * np.pi / wave_lengths
            wave_nos = wave_nos.reshape([1, 1, 1, -1])
        # ------------------------------------
        # phase delay indiced by height field
        # ------------------------------------

        # phi = height_map
            phase = K.cast(wave_nos * delta_N * height_map, K.float64)
            field = K.add(K.cast(K.cos(phase), dtype=K.complex64),
                                1.j * K.cast(K.sin(phase), dtype=K.complex64),
                                name='phase_plate_shift')

            input_shape = field.shape.as_list()
            [x, y] = np.mgrid[-input_shape[1] // 2: input_shape[1] // 2,
                     -input_shape[2] // 2: input_shape[2] // 2].astype(np.float64)

            max_val = np.amax(x)
            #max_val = int(self.patch_len / 2)

            r = np.sqrt(x ** 2 + y ** 2)[None, :, :, None]
            aperture = (r < max_val).astype(np.float64)
            output_field_circ = aperture * field

            N_psf = int((self.Nt * 2 * self.fac_m) * self.patch_size / self.wave_resolution[0])  # N_psf^2 number of psfs
            frac_N_Nl = self.patch_size / self.wave_resolution[0]  # decimation factor
            pix_shift = int((-np.floor(self.patch_size / N_psf / 2) + self.patch_size / N_psf / 2) * 2 * N_psf)

            for rec in range(N_psf):
                for rec2 in range(N_psf):
                    movi = int(pix_shift - rec * frac_N_Nl)
                    movj = int(pix_shift - rec2 * frac_N_Nl)
                    maskShif = K.roll(K.roll(Mask[:, :, band], movj, axis=1), movi, axis=0)
                    maskShif = K.expand_dims(K.expand_dims(K.cast(maskShif, dtype=K.complex64), 0), -1)
                    out_field1 = propagation(output_field_circ, self.sample_interval, wave_lengths, self.distance_code)
                    out_field1 = K.multiply(out_field1, maskShif)
                    out_field1 = propagation_back(out_field1, self.sample_interval, wave_lengths, self.distance_code)
                    out_field1 = propagation(out_field1, self.sample_interval, wave_lengths, self.distance)

                    psfs = K.square(K.abs(out_field1), name='intensities')
                    # Downsample psf to image resolution & normalize to sum to 1
                    psfs = area_downsampling_tf(psfs, self.patch_size)
                    psfs = K.math.divide(psfs, K.reduce_sum(psfs, axis=[1, 2]))

                    fac = int(self.patch_size / N_psf)
                    aux = np.zeros((N_psf, N_psf))
                    aux[rec, rec2] = 1
                    S = np.kron(np.ones((fac, fac)), aux)
                    S = K.expand_dims(K.convert_to_tensor(S, dtype=K.float32), 0)
                    psfs = K.transpose(psfs, [1, 2, 0, 3])
                    output_image = img_psf_conv(K.expand_dims(inputs[:, :, :, band] * S, -1), psfs)
                    if y_med_r is not None:
                        y_med_r = K.concat([y_med_r, self.fr[0, 0, band] * output_image], axis=3)
                    else:
                        y_med_r = self.fr[0, 0, band] * output_image

                    if y_med_g is not None:
                        y_med_g = K.concat([y_med_g, self.fg[0, 0, band] * output_image], axis=3)
                    else:
                        y_med_g = self.fg[0, 0, band] * output_image

                    if y_med_b is not None:
                        y_med_b = K.concat([y_med_b, self.fb[0, 0, band] * output_image], axis=3)
                    else:
                        y_med_b = self.fb[0, 0, band] * output_image

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



