#!python3
from numpy.core.overrides import array_function_dispatch, set_module
from numpy.fft import ifftshift
from numpy.core import integer, empty, arange, asarray, roll
import tensorflow as tf
import numpy as np
from scipy import interpolate

__all__ = ['fftshift', 'ifftshift', 'fftfreq', 'rfftfreq']

@tf.function
def propagation(output_field_circ, sample_interval, wave_lengths, distance):
    input_shape = output_field_circ.shape.as_list()

    _, M_orig, N_orig, _ = input_shape
    # zero padding.
    Mpad = M_orig // 4
    Npad = N_orig // 4
    M = M_orig + 2 * Mpad
    N = N_orig + 2 * Npad

    padded_input_field = tf.pad(output_field_circ,
                                [[0, 0], [Mpad, Mpad], [Npad, Npad], [0, 0]])

    [x, y] = np.mgrid[-N // 2:N // 2,
             -M // 2:M // 2]
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    # Spatial frequency
    fx = x / (sample_interval * N)  # max frequency = 1/(2*pixel_size)
    fy = y / (sample_interval * M)
    # We need to ifftshift fx and fy here, because ifftshift doesn't exist in TF.
    fx = tf.signal.ifftshift(fx)
    fy = tf.signal.ifftshift(fy)

    fx = fx[None, :, :, None]
    fy = fy[None, :, :, None]

    squared_sum = tf.square(fx) + tf.square(fy)

    constant_exponent_part = wave_lengths * np.pi * -1. * squared_sum * distance
    #constant_exponent_part = tf.convert_to_tensor(tmp, dtype=tf.float32)
    H = compl_exp_tf(constant_exponent_part, dtype=tf.complex64,
                         name='fresnel_kernel')


    objFT = transp_fft2d(padded_input_field)
    out_field1 = transp_ifft2d(objFT * H)
    out_field1 = out_field1[:, Mpad:-Mpad, Npad:-Npad, :]
    return out_field1

@tf.function
def propagation_back(output_field_circ, sample_interval, wave_lengths, distance):
    input_shape = output_field_circ.shape.as_list()

    _, M_orig, N_orig, _ = input_shape
    # zero padding.
    Mpad = M_orig // 4
    Npad = N_orig // 4
    M = M_orig + 2 * Mpad
    N = N_orig + 2 * Npad

    padded_input_field = tf.pad(output_field_circ,
                                [[0, 0], [Mpad, Mpad], [Npad, Npad], [0, 0]])

    [x, y] = np.mgrid[-N // 2:N // 2,
             -M // 2:M // 2]
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    # Spatial frequency
    fx = x / (sample_interval * N)  # max frequency = 1/(2*pixel_size)
    fy = y / (sample_interval * M)

    # We need to ifftshift fx and fy here, because ifftshift doesn't exist in TF.
    fx = tf.signal.ifftshift(fx)
    fy = tf.signal.ifftshift(fy)

    fx = fx[None, :, :, None]
    fy = fy[None, :, :, None]

    squared_sum = tf.square(fx) + tf.square(fy)


    constant_exponent_part = wave_lengths * np.pi * -1. * squared_sum * distance
    #constant_exponent_part = tf.convert_to_tensor(tmp, dtype=tf.float32)
    H = compl_exp_tf(constant_exponent_part, dtype=tf.complex64,
                         name='fresnel_kernel')

    H = tf.math.conj(H)
    objFT = transp_fft2d(padded_input_field)
    out_field1 = transp_ifft2d(objFT * H)
    out_field1 = out_field1[:, Mpad:-Mpad, Npad:-Npad, :]
    return out_field1

@tf.function
def kronecker_product(mat1, mat2):
  """Computes the Kronecker product two matrices."""
  m1, n1 = mat1.get_shape().as_list()
  mat1_rsh = tf.reshape(mat1, [m1, 1, n1, 1,1])
  m2, n2,l1 = mat2.get_shape().as_list()
  mat2_rsh = tf.reshape(mat2, [1, m2, 1, n2,l1])
  return tf.reshape(tf.multiply(mat1_rsh, mat2_rsh), [m1 * m2, n1 * n2,l1])

#@tf.function
def get_color_bases(wls):
    SG = [0.0875,0.1098,0.1157,0.1245,0.1379,0.1561,0.1840,0.2458,0.3101,0.3384,0.3917\
        ,0.5000,0.5732,0.6547,0.6627,0.624,0.5719,0.5157,0.4310,0.3470,0.2670,0.1760\
        ,0.1170,0.0874,0.0754,0.0674,0.0667,0.0694,0.0567,0.0360,0.0213]   # green color

    SB = [0.2340,0.2885,0.4613,0.5091,0.5558,0.5740,0.6120,0.6066,0.5759,0.4997\
        ,0.4000,0.3000,0.2070,0.1360,0.0921,0.0637,0.0360,0.0205,0.0130,0.0110\
        ,0.0080,0.0060,0.0062,0.0084,0.0101,0.0121,0.0180,0.0215,0.0164,0.0085\
        ,0.0050]                                                           # blue color

    SR = [0.1020,0.1020,0.0790,0.0590,0.0460,0.0360,0.0297,0.0293,0.0310,0.03230\
          ,0.0317,0.0367,0.0483,0.0667,0.0580,0.0346,0.0263,0.0487,0.1716,0.4342\
          ,0.5736,0.5839,0.5679,0.5438,0.5318,0.5010,0.4810,0.4249,0.2979,0.1362\
          ,0.0651]                                                         # red color

    SC = [0.1895,0.2118,0.1947,0.1835,0.1839,0.1921,0.2137,0.2751,0.3411,0.3707\
        ,0.4234,0.5367,0.6215,0.7214,0.7207,0.6586,0.5982,0.5644,0.6026,0.7812\
        ,0.8406,0.7599,0.6849,0.6312,0.6072,0.5684,0.5477,0.4943,0.3546,0.1722\
        ,0.0864]                                                          # cyan color

    x_wvls = np.linspace(400e-9,700e-9,len(SG))

    fg = interpolate.interp1d(x_wvls, SG)
    fr = interpolate.interp1d(x_wvls, SR)
    fc = interpolate.interp1d(x_wvls, SC)
    fb = interpolate.interp1d(x_wvls, SB)

    fr = fr(wls)
    fg = fg(wls)
    fc = fc(wls)
    fb = fb(wls)

    fr = tf.convert_to_tensor(fr, dtype=tf.float32)
    fg = tf.convert_to_tensor(fg, dtype=tf.float32)
    fc = tf.convert_to_tensor(fc, dtype=tf.float32)
    fb = tf.convert_to_tensor(fb, dtype=tf.float32)

    fr = tf.expand_dims(tf.expand_dims(fr, 0), 0)
    fg = tf.expand_dims(tf.expand_dims(fg, 0), 0)
    fc = tf.expand_dims(tf.expand_dims(fc, 0), 0)
    fb = tf.expand_dims(tf.expand_dims(fb, 0), 0)


    return fr,fg,fc,fb


def least_common_multiple(a, b):
    return abs(a * b) / np.math.gcd(a, b) if a and b else 0

@tf.function
def area_downsampling_tf(input_image, target_side_length):
    input_shape = input_image.shape.as_list()
    input_image = tf.cast(input_image, tf.float32)

    if not input_shape[1] % target_side_length:
        factor = int(input_shape[1] / target_side_length)
        output_img = tf.nn.avg_pool(input_image,
                                    [1, factor, factor, 1],
                                    strides=[1, factor, factor, 1],
                                    padding="VALID")
    else:
        # We upsample the image and then average pool
        lcm_factor = least_common_multiple(target_side_length, input_shape[1]) / target_side_length

        if lcm_factor > 10:
            print(
                "Warning: area downsampling is very expensive and not precise if source and target wave length have a large least common multiple")
            upsample_factor = 10
        else:
            upsample_factor = int(lcm_factor)
        img_upsampled = tf.image.resize(input_image, size=2 * [upsample_factor * target_side_length])
        # img_upsampled = tf.image.resize_nearest_neighbor(input_image,
        #                                                size=2 * [upsample_factor * target_side_length])
        output_img = tf.nn.avg_pool(img_upsampled,
                                    [1, upsample_factor, upsample_factor, 1],
                                    strides=[1, upsample_factor, upsample_factor, 1],
                                    padding="VALID")

    return output_img

@tf.function
def transp_fft2d(a_tensor, dtype=tf.complex64):
    """Takes images of shape [batch_size, x, y, channels] and transposes them
    correctly for tensorflows fft2d to work.
    """
    # Tensorflow's fft only supports complex64 dtype
    a_tensor = tf.cast(a_tensor, tf.complex64)
    # Tensorflow's FFT operates on the two innermost (last two!) dimensions
    a_tensor_transp = tf.transpose(a_tensor, [0, 3, 1, 2])
    a_fft2d = tf.signal.fft2d(a_tensor_transp)
    a_fft2d = tf.cast(a_fft2d, dtype)
    a_fft2d = tf.transpose(a_fft2d, [0, 2, 3, 1])
    return a_fft2d

@tf.function
def transp_ifft2d(a_tensor, dtype=tf.complex64):
    a_tensor = tf.transpose(a_tensor, [0, 3, 1, 2])
    a_tensor = tf.cast(a_tensor, tf.complex64)
    a_ifft2d_transp = tf.signal.ifft2d(a_tensor)
    # Transpose back to [batch_size, x, y, channels]
    a_ifft2d = tf.transpose(a_ifft2d_transp, [0, 2, 3, 1])
    a_ifft2d = tf.cast(a_ifft2d, dtype)
    return a_ifft2d

@tf.function
def compl_exp_tf(phase, dtype=tf.complex64, name='complex_exp'):
    """Complex exponent via euler's formula, since Cuda doesn't have a GPU kernel for that.
    Casts to *dtype*.
    """
    phase = tf.cast(phase, tf.float64)
    return tf.add(tf.cast(tf.cos(phase), dtype=dtype),
                  1.j * tf.cast(tf.sin(phase), dtype=dtype),
                  name=name)




'''
def fftfreq(n, d=1.0):
    """
    Return the Discrete Fourier Transform sample frequencies.

    The returned float array `f` contains the frequency bin centers in cycles
    per unit of the sample spacing (with zero at the start).  For instance, if
    the sample spacing is in seconds, then the frequency unit is cycles/second.

    Given a window length `n` and a sample spacing `d`::

      f = [0, 1, ...,   n/2-1,     -n/2, ..., -1] / (d*n)   if n is even
      f = [0, 1, ..., (n-1)/2, -(n-1)/2, ..., -1] / (d*n)   if n is odd

    Parameters
    ----------
    n : int
        Window length.
    d : scalar, optional
        Sample spacing (inverse of the sampling rate). Defaults to 1.

    Returns
    -------
    f : ndarray
        Array of length `n` containing the sample frequencies.
    """
    if not isinstance(n, integer_types):
        raise ValueError("n should be an integer")
    val = 1.0 / (n * d)
    results = empty(n, int)
    N = (n - 1) // 2 + 1
    p1 = arange(0, N, dtype=int)
    results[:N] = p1
    p2 = arange(-(n // 2), 0, dtype=int)
    results[N:] = p2
    return results * val


@set_module('numpy.fft')
def rfftfreq(n, d=1.0):
    """
    Return the Discrete Fourier Transform sample frequencies
    (for usage with rfft, irfft).

    The returned float array `f` contains the frequency bin centers in cycles
    per unit of the sample spacing (with zero at the start).  For instance, if
    the sample spacing is in seconds, then the frequency unit is cycles/second.

    Given a window length `n` and a sample spacing `d`::

      f = [0, 1, ...,     n/2-1,     n/2] / (d*n)   if n is even
      f = [0, 1, ..., (n-1)/2-1, (n-1)/2] / (d*n)   if n is odd

    Unlike `fftfreq` (but like `scipy.fftpack.rfftfreq`)
    the Nyquist frequency component is considered to be positive.

    Parameters
    ----------
    n : int
        Window length.
    d : scalar, optional
        Sample spacing (inverse of the sampling rate). Defaults to 1.

    Returns
    -------
    f : ndarray
        Array of length ``n//2 + 1`` containing the sample frequencies.


    """
    if not isinstance(n, integer_types):
        raise ValueError("n should be an integer")
    val = 1.0 / (n * d)
    N = n // 2 + 1
    results = arange(0, N, dtype=int)
    return results * val


def fftshift2d_tf(a_tensor):
    input_shape = a_tensor.shape.as_list()

    new_tensor = a_tensor
    for axis in range(1, 3):
        split = (input_shape[axis] + 1) // 2
        mylist = np.concatenate((np.arange(split, input_shape[axis]), np.arange(split)))
        new_tensor = tf.gather(new_tensor, mylist, axis=axis)
    return new_tensor


def ifftshift2d_tf(a_tensor):
    input_shape = a_tensor.shape.as_list()

    new_tensor = a_tensor
    for axis in range(1, 3):
        n = input_shape[axis]
        split = n - (n + 1) // 2
        mylist = np.concatenate((np.arange(split, n), np.arange(split)))
        new_tensor = tf.gather(new_tensor, mylist, axis=axis)
    return new_tensor'''

@tf.function
def psf2otf(input_filter, output_size):
    '''Convert 4D tensorflow filter into its FFT.

    :param input_filter: PSF. Shape (height, width, num_color_channels, num_color_channels)
    :param output_size: Size of the output OTF.
    :return: The otf.
    '''
    # pad out to output_size with zeros
    # circularly shift so center pixel is at 0,0
    fh, fw, _, _ = input_filter.shape.as_list()

    if output_size[0] != fh:
        pad = (output_size[0] - fh) / 2

        if (output_size[0] - fh) % 2 != 0:
            pad_top = pad_left = tf.ceil(pad)
            pad_bottom = pad_right = tf.floor(pad)
        else:
            pad_top = pad_left = tf.cast(pad,dtype=tf.int16) + 1
            pad_bottom = pad_right = tf.cast(pad,dtype=tf.int16) - 1

        padded = tf.pad(input_filter, [[pad_top, pad_bottom],
                                       [pad_left, pad_right], [0, 0], [0, 0]], "CONSTANT")
    else:
        padded = input_filter

    padded = tf.transpose(padded, [2, 0, 1, 3])
    padded = tf.signal.ifftshift(padded)
    padded = tf.transpose(padded, [1, 2, 0, 3])

    ## Take FFT
    tmp = tf.transpose(padded, [2, 3, 0, 1])
    tmp = tf.signal.fft2d(tf.complex(tmp, 0.))
    return tf.transpose(tmp, [2, 3, 0, 1])

@tf.function
def img_psf_conv(img, psf, otf=None, adjoint=False, circular=False):
    '''Performs a convolution of an image and a psf in frequency space.

    :param img: Image tensor.
    :param psf: PSF tensor.
    :param otf: If OTF is already computed, the otf.
    :param adjoint: Whether to perform an adjoint convolution or not.
    :param circular: Whether to perform a circular convolution or not.
    :return: Image convolved with PSF.
    '''
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    psf = tf.convert_to_tensor(psf, dtype=tf.float32)

    img_shape = img.shape.as_list()

    if not circular:
        target_side_length = 2 * img_shape[1]

        height_pad = (target_side_length - img_shape[1]) / 2
        width_pad = (target_side_length - img_shape[1]) / 2

        pad_top, pad_bottom = int(np.ceil(height_pad)), int(np.floor(height_pad))
        pad_left, pad_right = int(np.ceil(width_pad)), int(np.floor(width_pad))

        img = tf.pad(img, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], "CONSTANT")
        img_shape = img.shape.as_list()

    img_fft = transp_fft2d(img)

    if otf is None:
        otf = psf2otf(psf, output_size=img_shape[1:3])
        otf = tf.transpose(otf, [2, 0, 1, 3])

    otf = tf.cast(otf, tf.complex64)
    img_fft = tf.cast(img_fft, tf.complex64)

    if adjoint:
        result = transp_ifft2d(img_fft * tf.math.conj(otf))
    else:
        result = transp_ifft2d(img_fft * otf)

    result = tf.cast(tf.math.real(result), tf.float32)

    if not circular:
        result = result[:, pad_top:-pad_bottom, pad_left:-pad_right, :]

    return result

@tf.function
def deta(Lb):
    Lb = Lb / 1e-6
    IdLens = tf.math.sqrt(
        1 + ((0.6961663 * (Lb ** 2)) / ((Lb ** 2) - 0.0684043** 2)) + ((0.4079426 * (Lb ** 2)) / ((Lb ** 2) - 0.1162414**2)) + (
                    (0.8974794 * (Lb ** 2)) / ((Lb ** 2) - 9.896161**2)))
    #IdLens = 1.5375 + 0.00829045 * (Lb ** -2) - 0.000211046 * (Lb ** -4)
    IdAir = 1 + 0.05792105 / (238.0185 - Lb ** -2) + 0.00167917 / (57.362 - Lb ** -2)
    val = tf.abs(IdLens - IdAir)
    return 0.958*val


def reg_binary(tens):
    a = tf.square(tens)
    b = tf.square(1-tens)
    return 8*tf.sqrt(tf.reduce_sum(a*b) + tf.square(tf.reduce_sum(tf.square(tf.reduce_sum(tf.square(tens), axis=0) - 1))))
