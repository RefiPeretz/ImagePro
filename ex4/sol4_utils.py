import numpy as np
from scipy.signal import convolve2d, convolve
from scipy import ndimage
from scipy.misc import imread,imsave
from skimage.color import rgb2gray
from math import floor
import matplotlib.pyplot as plt
import os

#TODO do we want to use this kind of convolve?
def conv_der(im):
    """
    Calculate the magnitude of an image
    :param image:
    :return magnitude of an image
    """
    derv_X = convolve2d(im, np.array([1, 0, -1]).reshape(1,3), mode="same")
    drev_Y = convolve2d(im, np.array([[1], [0], [-1]]).reshape(3,1), mode="same")
    return derv_X, drev_Y


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



def gaus_1d(kernel_size):
    """
    Calculate 1d gaus kernel.
    :param kernel_size: size of the kernel we want.
    :return 1d kernel of desiered size.
    """
    gaus_kernel = np.array([1, 1])
    for i in range(kernel_size - 2):
        gaus_kernel = convolve(gaus_kernel, np.array([1, 1]), mode ='full')
    gaus_kernel = gaus_kernel.astype(np.float32)
    gaus_kernel /= np.sum(gaus_kernel)
    return gaus_kernel


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    Build gaussain pyramid. with max_levels levels using
    filter_vec as blur filter
    :param im: Image for the pyramid
    :param max_levels: Max levels of the pyramid
    :param filter_vec: The vector which represent the
    filter we want
    :return gaussain pyramid
    """

    filter_vec = gaus_1d(filter_size).reshape(1,filter_size)
    pyr = []
    pyr.append(im)
    for i in range(max_levels - 1):
        if(im.shape[0] <= 16 or im.shape[1] <= 16):
            break

        im = ndimage.filters.convolve(im, filter_vec.T, mode='mirror')
        im = ndimage.filters.convolve(im, filter_vec, mode='mirror')

        im = im[::2, ::2]
        pyr.append(im)

    return [pyr,filter_vec]


def expand_im(im,filter_vec):
    """
    Expand image in a power of two. First zeropad
    and than blur
    :param im: Image to expand
    :param filter_vec: The vector which represent the
    filter we want
    :return expand image
    """
    rowSize , colSize = im.shape[0],im.shape[1]
    #zero pad image
    exp_img = np.zeros((2*rowSize, 2*colSize), dtype=np.float32)
    exp_img[::2, ::2] = im
    exp_img = ndimage.filters.convolve(exp_img, 2 * filter_vec.T, mode='mirror')
    return ndimage.filters.convolve(exp_img, 2 * filter_vec, mode='mirror')


def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    Build laplacian pyramid. with max_levels levels using
    filter_vec as blur filter
    :param im: Image for the pyramid
    :param max_levels: Max levels of the pyramid
    :param filter_vec: The vector which represent the
    filter we want
    :return laplacian pyramid
    """
    filter_vec = gaus_1d(filter_size).reshape(1, filter_size)
    g_pyr = build_gaussian_pyramid(im, max_levels, filter_size)[0]
    l_pyr = []
    for i in range(len(g_pyr) - 1):
        l_im = g_pyr[i] - expand_im(g_pyr[i + 1], filter_vec)
        l_pyr.append(l_im)

    l_pyr.append(g_pyr[-1])
    return [l_pyr, filter_vec]


def laplacian_to_image(lpyr, filter_vec, coeff):
    """
    Recreate image form its laplacian image
    :param lpyr: Pyramid of the image
    :param filter_vec: The vector which represent the
    filter we used to create the pyramid image.
    :param coeff: List of constant for every
    level of the pyramid.
    :return The image we restored.
    """
    #TODO check size
    size_list = len(lpyr)
    for i in range(size_list):
        lpyr[i] *= coeff[i]

    resIm = lpyr[size_list-1]
    for i in range(size_list- 1,0,-1):
        resIm = expand_im(resIm,filter_vec)
        resIm += lpyr[i-1]

    return resIm


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """
    Blend to images using a given mask
    :param im1: Image to blend
    :param im2: Image to blend
    :param mask: Mask we use in the blending process
    :param max_levels: Max levels of the pyramid
    :param filter_size_im: The vector which represent the
    filter we want for image
    :param filter_size_mask: The vector which represent the
    filter we want for mask
    :return Blended image.
    """
    l_im1_pyr,filter_vec = build_laplacian_pyramid(im1,max_levels,filter_size_im)
    l_im2_pyr = build_laplacian_pyramid(im2,max_levels,filter_size_im)[0]
    g_mask_pyr = build_gaussian_pyramid(mask.astype(np.float32),max_levels,filter_size_mask)[0]

    l_out = []
    for i in range(len(l_im1_pyr)):
        tmp1 = (g_mask_pyr[i]*l_im1_pyr[i])
        tmp2 = (1 - g_mask_pyr[i])*l_im2_pyr[i]
        l_out.append(tmp1 + tmp2)

    return np.clip(laplacian_to_image(l_out,filter_vec,[1]*len(l_im1_pyr)),0,1)


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
