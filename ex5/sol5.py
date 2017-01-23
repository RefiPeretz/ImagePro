import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread as imread, imsave as imsave
from skimage.color import rgb2gray
import random


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


def load_dataset(filenames, batch_size, corruption_func, crop_size):
    ims_dic = {}
    # while True:
















