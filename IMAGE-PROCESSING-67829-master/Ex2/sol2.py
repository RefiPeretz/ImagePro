import numpy as np
from scipy import signal as sig
from scipy.misc import imread as imread
from skimage.color import rgb2gray

REPRESENTATION_GRAYSCALE = 1
REPRESENTATION_RGB = 2
NUM_OF_PIXELS = 256
DEPTH = 2
RGB = 3
GRAY = 2

def read_image(filename, representation):
    """
    Reads a given image file and converts it into a given representation.
    :param filename: string containing the image filename to read.
    :param representation: representation code, either 1 or 2 defining if the output should be either a grayscale
                           image (1) or an RGB image (2)
    :return: An output image is represented by a matrix of class np.float32 with intensities
             (either grayscale or RGB channel intensities) normalized to the range [0, 1].
    """
    im = imread(filename)  # Load image
    if (representation == REPRESENTATION_GRAYSCALE) and (im.ndim == RGB):
        im = rgb2gray(im)
        im = im.astype(np.float32)
    elif (representation == REPRESENTATION_GRAYSCALE) and (im.ndim == GRAY):
        im = im.astype(np.float32)
    elif (representation == REPRESENTATION_RGB) and (im.ndim == RGB):
        im = im.astype(np.float32)
        im /= (NUM_OF_PIXELS - 1)  # normalize image between 0 to 1
    else:
        raise "Invalid representation or image format!"
    return im.astype(np.float32)


### Section 1.1 ###
def DFT(signal):
    """
    Transform a 1D discrete signal to its Fourier representation
    :param signal: array of dtype float32 with shape (N,1)
    :return: array of dtype complex128 with shape (N,1)
    """
    # if (np.max(signal) > 1.0) or (np.min(signal) < 0):
    #     raise "signal must be between 0 to 1"
    # if signal.dtype != np.float32:
    #     raise "signal must be float32 type"
    x = np.asarray(signal)
    N = x.shape[0]
    n = np.arange(N)
    u = n.reshape((N, 1))
    exp_M = np.exp(-2j * np.pi * u * n / N)
    return np.dot(exp_M, x)

def IDFT(fourier_signal):
    """
    Tramsform a 1D Fourier representation to its discrete signal form
    :param fourier_signal: array of dtype complex128 with shape (N,1)
    :return: array of dtype float32 with shape (N,1)
    """
    # if fourier_signal.dtype != np.complex128:
    #     raise "fourier_signal must be complex128 type"
    x = np.asarray(fourier_signal)
    N = x.shape[0]
    n = np.arange(N)
    u = n.reshape((N, 1))
    exp_M = np.exp(2j * np.pi * u * n / N)
    return np.real_if_close((1 / N) * np.dot(exp_M, x)).astype(np.complex128)


### Section 1.2 ###
def DFT2(image):
    """
    Convert a 2D discrete signal to its Fourier representation
    :param image: a grayscale image of dtype float32
    :return: a grayscale fourier representation of image
    """
    # check_image(image)
    return DFT(DFT(image).T).T

def IDFT2(fourier_image):
    """
    Converts a 2D Fourier representation to its discrete signal representation
    :param fourier_image: a grayscale fourier representation of an image
    :return: a grayscale discrete signal
    """
    # if fourier_image.ndim != 2:
    #     raise "image must be grayscale!"
    # if fourier_image.dtype != np.complex128:
    #     raise "image must be complex128 type"
    return np.real_if_close(IDFT(IDFT(fourier_image).T).T).astype(np.complex128)


### Section 2.1 ###
def conv_der(im):
    """
    Computes the magnitude of image derivatives in image space
    :param im: grayscale image of type float32
    :return: magnitude image of the input grayscale image in type float32
    """
    # check_image(im)
    vertical_kernel = np.array([[1., 0., -1.]])
    horizontal_kernel = np.array([[1.], [0.], [-1.]])
    dx = sig.convolve2d(im, vertical_kernel, mode='same')
    dy = sig.convolve2d(im, horizontal_kernel, mode='same')
    return (np.sqrt(np.abs(dx)**2 + np.abs(dy)**2)).astype(np.float32)


### Section 2.2 ###
def fourier_der(im):
    """
    Computes the magnitude of image derivatives in Fourier space
    :param im: grayscale image of type float32
    :return: magnitude image of the input grayscale image in type float32
    """
    # check_image(im)
    # x derivative
    N = im.shape[1]
    im_dx = np.fft.fftshift(DFT2(im)) * np.arange(np.floor(-1*N/2), np.floor(N/2), 1)
    im_dx = IDFT2(np.fft.ifftshift(im_dx)) * (2 * np.pi * 1j / N)
    # y derivative
    M = im.shape[0]
    im_dy = (np.fft.fftshift(DFT2(im)).T * np.arange(np.floor(-1*M/2), np.floor(M/2), 1)).T
    im_dy = IDFT2(np.fft.ifftshift(im_dy)) * (2 * np.pi * 1j / M)
    return (np.sqrt(np.abs(im_dx)**2 + np.abs(im_dy)**2)).astype(np.float32)


### Section 3.1 ###
def create_gaussian_kernel(size):
    """
    Creates a 2D squared matrix of binomial coefficients as an approximation to the gaussian distribution
    :param size: size of matrix side, integer
    :return: 2D squared matrix of of binomial coefficients normalized to total sum value of 1
    """
    base_kernel = np.array([[1., 1.]])
    res = base_kernel.copy()
    for i in range(size - 2):
        res = sig.convolve2d(res, base_kernel)
    res = sig.convolve2d(res, res.T)
    return res / np.sum(res)

def blur_spatial(im, kernel_size):
    """
    Performs image blurring using 2D convolution between the image and a gaussian kernel
    :param im: input image to be blurred (grayscale float32 image)
    :param kernel_size: size of the gaussian kernel in each dimension (an odd integer)
    :return: the output blurry image (grayscale float32 image)
    """
    # check_image(im)
    if (kernel_size % 2 == 0):
        raise "kernel size must be an odd integer!"
    gauss_kernel = create_gaussian_kernel(kernel_size)
    return sig.convolve2d(im,gauss_kernel,mode='same',boundary='wrap').astype(np.float32)


### Section 3.2 ###
def blur_fourier(im, kernel_size):
    """
    Performs image blurring with gaussian kernel in Fourier space
    :param im: input image to be blurred (grayscale float32 image)
    :param kernel_size: the size of the gaussian in each dimension (an odd integer)
    :return: the output blurry image (grayscale float32 image)
    """
    # check_image(im)
    if (kernel_size % 2 == 0):
        raise "kernel size must be an odd integer!"
    kernel_center_y = np.floor(im.shape[0]/2).astype(np.int)
    kernel_center_x = np.floor(im.shape[1]/2).astype(np.int)
    gauss_kernel = np.zeros(im.shape)
    gauss_kernel[kernel_center_y - int((kernel_size - 1)/2): kernel_center_y + int((kernel_size - 1)/2) + 1
                , kernel_center_x - int((kernel_size - 1)/2): kernel_center_x + int((kernel_size - 1)/2) + 1] \
                = create_gaussian_kernel(kernel_size)
    gauss_kernel = np.fft.ifftshift(gauss_kernel)
    return np.real(IDFT2(DFT2(im) * DFT2(gauss_kernel))).astype(np.float32)