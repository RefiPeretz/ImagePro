import numpy as np
from scipy.signal import convolve2d, convolve
from scipy.misc import imread as imread
from skimage.color import rgb2gray
from math import floor



def normlized_image(image):
    """
    Normlize image to float 32 and [0,1]
    :param image: Reprentaion of display 1 for grayscale 2 for RGB
    :return normlized image
    """
    if(image.dtype != np.float32):
        image = image.astype(np.float32)
    if(image.max() > 1):
        image /= 255

    return image


def is_rgb(im):
    """
    Verify if an image is RGB
    :param im: Reprentaion of display 1 for grayscale 2 for RGB
    """
    if(im.ndim == 3):
        return True
    else:
        return False


def validate_representation(representation):
    """
    Validate reprentaion input
    :param representation: Reprentaion of display 1 for grayscale 2 for RGB
    """

    if representation != 1 and representation != 2:
        raise Exception("Unkonwn representaion")


def read_image(fileame, representation):
    """
    Read image by file name and normlize it to float 32 [0,1] representaion
    according to RGB or Graysacle
    :param filename: The name of the file that we should read.
    :param representation: Reprentaion of display 1 for grayscale 2 for RGB
    :return normlized image
    """
    validate_representation(representation)

    im = imread(fileame)
    if representation == 1 and is_rgb(im):
        # We should convert from Grayscale to RGB
        im =rgb2gray(im)
        return im.astype(np.float32)

    return normlized_image(im)


def calc_matrix(size,exp_shape):
    """
    Calculate vandermonde matrix for size and type(DFT/IDFT)
    :param size: The size of the matrix
    :param exp_shape: The type of the matrix DFT/IDFT
    :return sizeXsize vandermonde matrix.
    """
    N = size
    x = np.arange(N)
    u = np.arange(N).reshape(N, 1)
    return np.exp(-2j * (np.pi * u * x / N)) if exp_shape == 1 else np.exp(2j * (np.pi * u * x / N))

def DFT(signal):
    """
    Calculate the DFT of the signal
    :param signal: Signal to transform.
    :return transformed signal
    """
    work_sig = np.copy(signal)
    w_matrix = calc_matrix(signal.shape[0],1)
    return np.dot(w_matrix, work_sig).astype(np.complex128)

def IDFT(signal):
    """
    Calculate the IDFT of the signal
    :param signal: Signal to transform.
    :return transformed signal.
    """
    work_sig = np.copy(signal)
    N = signal.shape[0]
    w_matrix = calc_matrix(N,2)
    return (np.dot(w_matrix, work_sig)/N)


def DFT2(image):
    """
    Calculate the DFT of an image
    :param image:
    :return transformed image
    """
    return np.dot(DFT(image), calc_matrix(image.shape[1],1)).astype(np.complex128)

def IDFT2(image):
    """
    Calculate the DFT of an image
    :param image:
    :return transformed image
    """
    return np.dot(IDFT(image), calc_matrix(image.shape[1],2)) / image.shape[1]

def conv_der(im):
    """
    Calculate the magnitude of an image
    :param image:
    :return magnitude of an image
    """
    derv_X = convolve2d(im, np.array([1, 0, -1]).reshape(1,3), mode="same")
    drev_Y = convolve2d(im, np.array([[1], [0], [-1]]).reshape(3,1), mode="same")
    return np.sqrt(np.power(derv_X, 2) + np.power(drev_Y, 2)).astype(np.float32)

def fourier_der(im):
    """
    Calculate the magnitude of an image by fourier transform.
    :param image:
    :return magnitude of an image ( by fourier transform)
    """
    im_DFT = DFT2(im)
    N_F, M_F, N_F_MINUS, M_F_MINUS, N , M = floor(im_DFT.shape[1]/2), floor(im_DFT.shape[0]/2), \
                                            floor(-1*(im_DFT.shape[1] / 2)),\
                                            floor(-1*(im_DFT.shape[0] / 2)),\
                                            im_DFT.shape[1], im_DFT.shape[0]
    u_Y = np.tile(np.concatenate((np.arange(0, M_F, 1), np.arange(M_F_MINUS, 0, 1))).reshape(M, 1), (1, N))
    u_X = np.tile(np.concatenate((np.arange(0,N_F,1),np.arange(N_F_MINUS,0,1))).reshape(1,N), (M,1))

    derv_X_dft = u_X * im_DFT
    derv_Y_dft =  u_Y * im_DFT
    derv_X, derv_Y = IDFT2((derv_X_dft) * (2j * np.pi / M)), IDFT2((derv_Y_dft) * (2j * np.pi / N))
    return (np.sqrt(np.abs(derv_X) ** 2 + np.abs(derv_Y) ** 2)).astype(np.float32)


def gaus_1d(kernel_size):
    """
    Calculate 1d gaus kernel.
    :param kernel_size: size of the kernel we want.
    :return 1d kernel of desiered size.
    """
    gaus_kernel = np.array([1, 1])
    for i in range(kernel_size - 2):
        gaus_kernel = convolve(gaus_kernel, np.array([1, 1]), mode ='full')
    return gaus_kernel

def gaus_2d(kernel_size):
    """
    Calculate 2d gaus kernel by 1d gaus kernel.
    :param kernel_size: size of the kernel we want.
    :return 2d kernel of desiered size.
    """
    d1_kernel = gaus_1d(kernel_size).reshape(1,kernel_size)
    return convolve2d(d1_kernel, d1_kernel.T, mode='full').astype(np.float32)

def blur_spatial(im, kernel_size):
    """
    Blur image in spatial using convolution with
     a given size gaus kernel
    :param im, image to blur
    :param kernel_size: size of the kernel we want.
    :return blur image.
    """
    gaus_kerenel = gaus_2d(kernel_size)
    gaus_kerenel /= np.sum(gaus_kerenel)
    return convolve2d(im, gaus_kerenel, mode='same', boundary='wrap')

def blur_fourier(im, kernel_size):
    """
    Blur image in fourier using dot with transformed
    kerenel in a given size gaus kernel
    :param im, image to blur
    :param kernel_size: size of the kernel we want.
    :return Blur image.
    """
    gaus_kerenel = gaus_2d(kernel_size)
    gaus_kerenel /= np.sum(gaus_kerenel)
    N,M = im.shape[0],im.shape[1]
    i, j = floor(N / 2), floor(M / 2)
    gaus_kernel_pad = np.zeros(shape=(N, M))
    low_x, top_x = i-floor(kernel_size/2), i + floor(kernel_size/2) + 1
    low_y, top_y = j-floor(kernel_size/2), j + floor(kernel_size/2) + 1

    gaus_kernel_pad[low_x:top_x, low_y:top_y] = gaus_kerenel


    gaus_kernel_pad = np.fft.fftshift(gaus_kernel_pad)
    im_DFT, gaus_kerenel_DFT = DFT2(im),  DFT2(gaus_kernel_pad)
    return IDFT2(im_DFT*gaus_kerenel_DFT).real.astype(np.float32)
