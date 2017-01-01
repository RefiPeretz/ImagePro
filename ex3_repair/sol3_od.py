import numpy as np
from skimage import color
from scipy import signal as sig
from scipy.misc import imread as imread
from scipy.ndimage.filters import convolve
import os
import matplotlib.pyplot as plt

SMALLEST_PYRAMID_SIZE = 16
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
    im = imread(filename)
    if representation == 1 and im.ndim == 3 and im.shape[2] == 3:
        im = color.rgb2gray(im)
    if im.dtype == np.uint8:
        im = im.astype(np.float32) / 255.0
    return im


### Section 3.1 ###
def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    Construct a Gaussian pyramid.
    :param im: a grayscale image with double values in [0,1].
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: he size of the Gaussian filter (an odd scalar that represents a squared filter) to be used
                        in constructing the pyramid filter.
    :return: A tuple with the gaussian pyramid and the filter vector used to make the pyramid.
    """
    if filter_size % 2 == 0:
        raise "kernel size must be an odd integer!"

    filter_vec = create_gaussian_kernel(filter_size, 1)
    pyr = [im]
    next_level_size_x = im.shape[1] / 2
    next_level_size_y = im.shape[0] / 2
    current_level_counter = 1
    im_work = np.copy(im)

    while ((current_level_counter < max_levels)
        and (next_level_size_x >= SMALLEST_PYRAMID_SIZE)
            and (next_level_size_y >= SMALLEST_PYRAMID_SIZE)):

        im_work = reduce(im_work, filter_vec)
        pyr.append(im_work)

        next_level_size_x = im_work.shape[1] / 2
        next_level_size_y = im_work.shape[0] / 2
        current_level_counter += 1

    return pyr, filter_vec


def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    Construct a Laplacian pyramid.
    :param im: a grayscale image with double values in [0,1].
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: he size of the Gaussian filter (an odd scalar that represents a squared filter) to be used
                        in constructing the pyramid filter.
    :return: A tuple with the laplacian pyramid and the filter vector used to make the pyramid.
    """
    if filter_size % 2 == 0:
        raise "kernel size must be an odd integer!"

    gaussian_pyr, filter_vec = build_gaussian_pyramid(np.copy(im), max_levels, np.copy(filter_size))
    laplacian_pyr = []

    i = 0
    for i in range(len(gaussian_pyr) - 1):
        laplacian_pyr.append(gaussian_pyr[i] - expand(gaussian_pyr[i+1], filter_vec))
    laplacian_pyr.append(gaussian_pyr[i+1])

    return laplacian_pyr, filter_vec


def reduce(im, filter_vec):
    """
    Reduces image size by half using blur and subsampling.
    :param im: image to reduce.
    :param filter_vec: filter to use for blurring.
    :return: Reduced image.
    """
    im_work = np.copy(im)
    # blur stage
    im_work = convolve(im_work, filter_vec, mode='mirror')
    im_work = convolve(im_work, filter_vec.T, mode='mirror')
    # subsample stage
    return im_work[::2, ::2]


def expand(im, filter_vec):
    """
    Expaned image size to twice its size using padding and blur.
    :param im: image to expand.
    :param filter_vec: filter to use for blurring.
    :return: Expanded image.
    """
    filter_vec_work = np.copy(filter_vec)
    # expand stage
    im_work = np.zeros((2 * im.shape[0], 2 * im.shape[1]), dtype=np.float32)
    im_work[::2, ::2] = im
    # blur stage
    filter_vec_work *= 2
    im_work = convolve(im_work, filter_vec_work, mode='mirror')
    im_work = convolve(im_work, filter_vec_work.T, mode='mirror')
    return im_work


def create_gaussian_kernel(size, dimensions):
    """
    Creates a 1D/2D squared matrix of binomial coefficients as an approximation to the gaussian distribution
    :param size: size of matrix side, integer
    :param dimensions: selector for 1D or 2D representation.
    :return: 1D/2D squared matrix of of binomial coefficients normalized to total sum value of 1
    """
    #if size == 1:
    #    return np.array([[0]])
    base_kernel = np.array([[1., 1.]])
    res = base_kernel.copy()
    for i in range(size - 2):
        res = sig.convolve2d(res, base_kernel)
    if dimensions == 2:
        res = sig.convolve2d(res, res.T)
    return res / np.sum(res)


### Section 3.2 ###
def laplacian_to_image(lpyr, filter_vec, coeff):
    """
    Performs reconstruction of an image from its Laplacian Pyramid.
    :param lpyr: the Laplacian pyramid that is generated by the function in 3.1.
    :param filter_vec: the filter vector that is generated by the function in 3.1.
    :param coeff: coefficients vector.
    :return: The reconstructed image.
    """
    coeff = np.array(coeff)
    lpyr = lpyr * coeff
    rec_im = lpyr[-1] # == G_n
    n = len(lpyr) - 1

    for lvl in range(n, 0, -1):
        rec_im = expand(rec_im, filter_vec) + lpyr[lvl - 1]
    return rec_im


### Section 3.3 ###
def render_pyramid(pyr, levels):
    """
    Builds a horizontal appending of the given pyramid images.
    :param pyr: given gaussian or laplacian pyramid.
    :param levels: number of level to appand from the pyramid.
    :return: the appended image.
    """
    res = image_stretch(pyr[0])
    if levels > len(pyr):
        levels = len(pyr)
    for lvl in range(1, levels, 1):
        im_to_append = np.pad(image_stretch(pyr[lvl]),
                              ((0, res.shape[0] - pyr[lvl].shape[0]), (0, 0)), 'constant', constant_values=0)
        res = np.append(res, im_to_append, axis=1)
    return res.astype(np.float32)


def display_pyramid(pyr, levels):
    """
    Displays render_pyramid output.
    :param pyr: given gaussian or laplacian pyramid.
    :param levels: number of level to appand from the pyramid.
    :return: void.
    """
    res = render_pyramid(pyr, levels)
    plt.figure()
    plt.imshow(res, cmap=plt.cm.gray)


def image_stretch(image):
    """
    Stretches image between 0 to 1.
    :param image: image to stretch.
    :return: Starched image.
    """
    image_work = np.copy(image)
    image_work = ((image_work - image_work.min()) / (image_work.max() - image_work.min())).astype(np.float32)
    return image_work


### Section 4.0 ###
def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """
    Implement pyramid blending as described in the lecture. Note that im1, im2 and mask should all have the same
    dimensions and that once again you can assume that image dimensions are multiples of 2 (max_levels−1).
    :param im1: input grayscale image to be blended.
    :param im2: input grayscale images to be blended.
    :param mask: is a boolean (i.e. dtype == np.bool) mask containing True and False representing which parts
                 of im1 and im2 should appear in the resulting im_blend. Note that a value of True corresponds to 1,
                 and False corresponds to 0.
    :param max_levels: is the max_levels parameter you should use when generating the Gaussian and Laplacian
                       pyramids.
    :param filter_size_im: is the size of the Gaussian filter (an odd scalar that represents a squared filter) which
                           defining the filter used in the construction of the Laplacian pyramids of im1 and im2.
    :param filter_size_mask: is the size of the Gaussian filter(an odd scalar that represents a squared filter) which
                             defining the filter used in the construction of the Gaussian pyramid of mask.
    :return: The blended image.
    """

    l_pyr_im1, filter_vec = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    l_pyr_im2 = build_laplacian_pyramid(im2, max_levels, filter_size_im)[0]
    g_pyr_mask = build_gaussian_pyramid(mask.astype(np.double), max_levels, filter_size_mask)[0]

    l_pyr_out = []
    for lvl in range(len(l_pyr_im1)):
        l_pyr_out.append(g_pyr_mask[lvl] * l_pyr_im1[lvl] + (1 - g_pyr_mask[lvl]) * l_pyr_im2[lvl])

    im_out = laplacian_to_image(l_pyr_out, filter_vec, np.ones(len(l_pyr_out), dtype=np.float32))
    return im_out.clip(min=0.0, max=1.0)


### Section 4.1 ###
def relpath(filename):
    """
    Helper function to read a file.
    :param filename: file to read.
    :return: the file that was read.
    """
    return os.path.join(os.path.dirname(__file__), filename)


def blending_example1():
    """
    Performs pyramid blending on a set of image pairs and mask. Prints the images, mask and blended
    image to the screen.
    :return: The blended image.
    """
    im1 = read_image(relpath('external/soup_1024_1024.jpg'), REPRESENTATION_RGB)
    im2 = read_image(relpath('external/waterpollo_1024_1024.jpg'), REPRESENTATION_RGB)
    mask = read_image(relpath('external/waterpollo_mask_1024_1024.jpg'), REPRESENTATION_GRAYSCALE)
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    mask = mask.astype(np.bool)

    res = np.zeros(im1.shape)

    res[:, :, 0] = pyramid_blending(im1[:, :, 0], im2[:, :, 0], mask, 2, 3, 3)
    res[:, :, 1] = pyramid_blending(im1[:, :, 1], im2[:, :, 1], mask, 2, 3, 3)
    res[:, :, 2] = pyramid_blending(im1[:, :, 2], im2[:, :, 2], mask, 2, 3, 3)

    res = res.astype(np.float32)

    plt.figure("blending_example1")
    plt.subplot(2, 2, 1, title="im1")
    plt.imshow(im1)
    plt.subplot(2, 2, 2, title="im2")
    plt.imshow(im2)
    plt.subplot(2, 2, 3, title="mask")
    plt.imshow(mask, cmap=plt.cm.gray)
    plt.subplot(2, 2, 4, title="blended")
    plt.imshow(res)

    return im1, im2, mask, res


def blending_example2():
    """
    Performs pyramid blending on a set of image pairs and mask. Prints the images, mask and blended
    image to the screen.
    :return: The blended image.
    """
    im1 = read_image(relpath('external/gal_1024_1024.jpg'), REPRESENTATION_RGB)
    im2 = read_image(relpath('external/spider_1024_1024.jpg'), REPRESENTATION_RGB)
    mask = read_image(relpath('external/spider_mask_1024_1024.jpg'), REPRESENTATION_GRAYSCALE)
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    mask = mask.astype(np.bool)

    res = np.zeros(im1.shape)

    res[:, :, 0] = pyramid_blending(im1[:, :, 0], im2[:, :, 0], mask, 4, 3, 3)
    res[:, :, 1] = pyramid_blending(im1[:, :, 1], im2[:, :, 1], mask, 4, 3, 3)
    res[:, :, 2] = pyramid_blending(im1[:, :, 2], im2[:, :, 2], mask, 4, 3, 3)

    res = res.astype(np.float32)

    plt.figure("blending_example2")
    plt.subplot(2, 2, 1, title="im1")
    plt.imshow(im1)
    plt.subplot(2, 2, 2, title="im2")
    plt.imshow(im2)
    plt.subplot(2, 2, 3, title="mask")
    plt.imshow(mask, cmap=plt.cm.gray)
    plt.subplot(2, 2, 4, title="blended")
    plt.imshow(res)

    return im1, im2, mask, res
