import numpy as np
from scipy.misc import imread as imread
from skimage import color
from scipy.ndimage.filters import convolve
from scipy.signal import convolve2d as convolve2d
import matplotlib.pyplot as plt
import os as os

MAX_PIXEL_VALUE = 255
IMG_GRAYSCALE = 1
IMG_RGB = 2
RGB_IMG_DIMENTIONS = 3
MAX_SHAPE_TO_REDUCE = 32
R = 0
G = 1
B = 2

def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    construct a Gaussian pyramid of a given image

    :param im: a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid
    :param filter_size: the size of the Gaussian filter (an odd scalar that represents a squared filter) to be used
           in constructing the pyramid filter
    :return: pyr, filter_vec

             pyr: resulting pyramid pyr as a standard python array (i.e. not numpy’s array) with maximum length of
             max_levels, where each element of the array is a grayscale image.

             filter_vec: 1D-row of size filter_size used for the pyramid construction
    """

    filter_vec = __get_gaussian_kernel(filter_size)

    G0 = im.copy()
    pyr = [G0]

    for level in range(max_levels - 1):
        if pyr[level].shape[0] < MAX_SHAPE_TO_REDUCE or pyr[level].shape[1] < MAX_SHAPE_TO_REDUCE:
            break
        pyr.append(reduce(pyr[level], filter_vec))

    return pyr, filter_vec


def build_laplacian_pyramid(im, max_levels, filter_size):
    """

    :param im: a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid
    :param filter_size: the size of the Gaussian filter (an odd scalar that represents a squared filter) to be used
           in constructing the pyramid filter
    :return: pyr, filter_vec

             pyr: resulting pyramid pyr as a standard python array (i.e. not numpy’s array) with maximum length of
             max_levels, where each element of the array is a grayscale image.

             filter_vec: 1D-row of size filter_size used for the pyramid construction
    """

    gaussian_pyr, filter_vec = build_gaussian_pyramid(im, max_levels, filter_size)

    l_pyr = []

    for level in range(len(gaussian_pyr)):
        if level == len(gaussian_pyr) - 1:
            l_pyr.append(gaussian_pyr[level])
        else:
            l_pyr.append(gaussian_pyr[level] - expand(gaussian_pyr[level+1], filter_vec))

    return l_pyr, filter_vec


def __get_gaussian_kernel(kernel_size):
    bin_coefficient = first_bin_coefficient = np.array([1., 1.]).reshape(1, 2)
    for i in range(kernel_size - 2):
        bin_coefficient = convolve2d(bin_coefficient, first_bin_coefficient)

    kernel = bin_coefficient / bin_coefficient.sum()  # normilizing

    return kernel


def laplacian_to_image(lpyr, filter_vec, coeff):
    """

    :param lpyr: the Laplacian pyramid
    :param filter_vec: 1D-row of size filter_size used for the pyramid construction
    :param coeff: A vector of coeeficients that are multiplying each level i of the laplacian pyramid by its
                  corresponding coecient coeff[i], The vector size is the same as the number of levels in the pyramid
                  lpyr.
    :return: im - the original image
    """
    lpyr = lpyr.copy()
    lpyr *= np.array(coeff)

    n = len(lpyr) - 1  # number of levels
    image = lpyr[n]    # init as deepest pyramid level

    for level in range(n, 0, -1):
        image = expand(image, filter_vec) + lpyr[level - 1]

    return image

def reduce(im, filter_vec):
    # blur rows and cols separately (for efficiency)
    res = convolve(im, filter_vec, mode='mirror')
    res = convolve(res, filter_vec.T, mode='mirror')

    # subsample every 2nd row of every second column
    res = res[::2, ::2]

    return res.astype(np.float32)


def expand(im, filter_vec):
    filter_vec = filter_vec.copy() * 2
    res = np.zeros(shape=(2*im.shape[0], 2*im.shape[1]))
    res[::2, ::2] = im.copy()
    res = convolve(res, filter_vec, mode='mirror')
    res = convolve(res, filter_vec.T, mode='mirror')

    return res.astype(np.float32)

def render_pyramid(pyr, levels):
    """

    :param pyr: is either a Gaussian or Laplacian pyramid
    :param levels: is the number of levels to present in the result ≤ max_levels
    :return: res
             res: is a single black image in which the pyramid levels of the
             given pyramid pyr are stacked horizontally (after stretching the values to [0, 1])
    """
    if levels > len(pyr):
        levels = len(pyr)

    res_height = pyr[0].shape[0]
    res_width = 0
    for lvl in range(levels):
        res_width += pyr[lvl].shape[1]

    res = np.zeros((res_height, res_width), dtype=np.double)

    location = 0
    for lvl in range(levels):
        res[0:pyr[lvl].shape[0], location:location + pyr[lvl].shape[1]] = stretch(pyr[lvl])
        location += pyr[lvl].shape[1]

    return res.astype(np.float32)

def display_pyramid(pyr, levels):
    res = render_pyramid(pyr, levels)
    fig = plt.figure()
    plt.imshow(res, cmap=plt.cm.gray)


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """

    :param im1: first input grayscale images to be blended
    :param im2: second input grayscale images to be blended
    :param mask: is a boolean (i.e. dtype == np.bool) mask containing True and False representing which parts
                 of im1 and im2 should appear in the resulting im_blend. Note that a value of True corresponds to 1,
                 and False corresponds to 0.
    :param max_levels: is the max_levels parameter you should use when generating the Gaussian and Laplacian pyramids.
    :param filter_size_im: is the size of the Gaussian filter (an odd scalar that represents a squared filter) which
                           defining the filter used in the construction of the Laplacian pyramids of im1 and im2.
    :param filter_size_mask: is the size of the Gaussian filter(an odd scalar that represents a squared filter) which
                             defining the filter used in the construction of the Gaussian pyramid of mask.
    :return: im_blend - the blended image
    """

    l_pyr_im1, im1_filter_vec = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    l_pyr2_im2, im2_filter_vec = build_laplacian_pyramid(im2, max_levels, filter_size_im)
    g_pyr_mask, mask_filter_vec = build_gaussian_pyramid(mask.astype(np.double), max_levels, filter_size_mask)
    l_pyr_out = []

    for level in range(len(l_pyr_im1)):
        l_pyr_out.append(g_pyr_mask[level] * l_pyr_im1[level] + (1 - g_pyr_mask[level]) * l_pyr2_im2[level])

    coeff = np.ones(len(l_pyr_out), dtype=np.double)
    im_blend = laplacian_to_image(l_pyr_out, im1_filter_vec, coeff)

    return im_blend.clip(0, 1)


def blending_example1():
    im1 = read_image(relpath('external/blending_example1_im1.jpg'), IMG_RGB)
    im2 = read_image(relpath('external/blending_example1_im2.jpg'), IMG_RGB)
    mask = (read_image(relpath('external/blending_example1_mask.jpg'), IMG_GRAYSCALE))
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    mask = mask.astype(np.bool)

    fig = plt.figure("blending_example1")
    fig.add_subplot(221, title="im1")
    plt.imshow(im1)
    fig.add_subplot(222, title="im2")
    plt.imshow(im2)
    fig.add_subplot(223, title="mask")
    plt.imshow(mask, cmap=plt.cm.gray)

    im_blend = np.zeros(im1.shape, dtype=np.float32)

    for channel in (R, G, B):
        im_blend[:, :, channel] = pyramid_blending(im1[:, :, channel], im2[:, :, channel], mask, 9, 3, 3)

    fig.add_subplot(224)
    plt.imshow(im_blend)

    return im1, im2, mask, im_blend


def blending_example2():
    im1 = read_image(relpath('external/blending_example2_im1.jpg'), IMG_RGB)
    im2 = read_image(relpath('external/blending_example2_im2.jpg'), IMG_RGB)
    mask = read_image(relpath('external/blending_example2_mask.jpg'), IMG_GRAYSCALE)
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    mask = mask.astype(np.bool)
    fig = plt.figure("blending_example2")
    fig.add_subplot(221, title="im1")
    plt.imshow(im1)
    fig.add_subplot(222, title="im2")
    plt.imshow(im2)
    fig.add_subplot(223, title="mask")
    plt.imshow(mask, cmap=plt.cm.gray)

    im_blend = np.zeros(im1.shape, dtype=np.float32)

    for channel in (R, G, B):
        im_blend[:, :, channel] = pyramid_blending(im1[:, :, channel], im2[:, :, channel], mask, 9, 5, 5)

    fig.add_subplot(224)
    plt.imshow(im_blend)

    return im1, im2, mask, im_blend


def stretch(im):
    """
    stretches values linarly to [0-1]
    :param im: image to stretch
    :return: stretched image
    """
    res = im - im.min()
    res /= res.max()
    return res


def relpath(filename):
    return os.path.join(os.path.dirname(__file__), filename)

def read_image(filename, representation):
    """Reads image from file name with either Grayscale or RGB represantation

    :param filename: string containing the image filename to read
    :param representation: representation code: grayscale image (1), RGB image (2).
    :return: the image with the chosen representation
    """
    im = imread(filename)
    if representation == IMG_GRAYSCALE and im.ndim == 3 and im.shape[2] == RGB_IMG_DIMENTIONS:
        im = color.rgb2gray(im)
    if im.dtype == np.uint8:
        im = im.astype(np.float32) / MAX_PIXEL_VALUE
    return im
