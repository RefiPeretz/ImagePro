import numpy as np
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from scipy.misc import imread as imread

# Defines:
RGB_TO_YIQ_TRANS_MATRIX = np.array([[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]],
                                   np.float32)
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
        im /= (NUM_OF_PIXELS - 1)  # normalize image between 0 to 1
    elif (representation == REPRESENTATION_RGB) and (im.ndim == RGB):
        im = im.astype(np.float32)
        im /= (NUM_OF_PIXELS - 1)  # normalize image between 0 to 1
    else:
        raise "Invalid representation or image format!"
    return im.astype(np.float32)


def imdisplay(filename, representation):
    """
    A function that utilizes read_image to display a given image file in a given representation.
    :param filename: string containing the image filename to read.
    :param representation: representation code, either 1 or 2 defining if the output should be either a grayscale
                           image (1) or an RGB image (2).
    """
    image = read_image(filename, representation)
    plt.figure()
    if representation == REPRESENTATION_GRAYSCALE:
        plt.imshow(image, cmap=plt.cm.gray)
    elif representation == REPRESENTATION_RGB:
        plt.imshow(image)
    plt.axis('off')


def rgb2yiq(imRGB):
    """
    Transform an YIQ image into the RGB color space.
    :param imRGB: eight×width×3 np.float32 matrices with values
                  in [0, 1]
    :return: eight×width×3 np.float32 matrices of RGB color space.
    """
    # check that the matrix is RGB and not greyscale = 3dims
    return (np.dot(imRGB, RGB_TO_YIQ_TRANS_MATRIX.T)).astype(np.float32)


def yiq2rgb(imYIQ):
    """
    Transform an RGB image into the YIQ color space.
    :param imRGB: eight×width×3 np.float32 matrices with values
                  in [0, 1]
    :return: eight×width×3 np.float32 matrices of YIQ color space.
    """
    # check that the matrix is YIQ and not greyscale = 3dims
    return (np.dot(imYIQ, np.linalg.inv(RGB_TO_YIQ_TRANS_MATRIX).T)).astype(np.float32)


def validate_and_prepare_image(im):
    """
    Checks if image is RGB or GRAY, if RGB will take Y channel only.
    Then normalize the image between 0 to 255 uint8.
    :param im: image to operate on
    :return: the normalized image.
    """
    if im.ndim == RGB:
        im = rgb2yiq(im)[:, :, 0]
        im = np.round(im * (NUM_OF_PIXELS - 1)).astype(np.uint8)
    elif im.ndim == GRAY:
        im = np.round(im * (NUM_OF_PIXELS - 1)).astype(np.uint8)
    else:
        raise "Invalid image format!"
    return im


def return_y_channel_to_rgb(y_channel, im_orig):
    """
    Replaces the Y channel of im_orig and reverts back to RGB image
    :param y_channel: 2d matrix of y_values
    :param im_orig: the image that its y channel will be replaced
    :return: the replaced image in RGB form
    """
    im_orig_tmp = rgb2yiq(im_orig)
    im_orig_tmp[:, :, 0] = y_channel.astype(np.float32) / (NUM_OF_PIXELS - 1)
    return np.clip(yiq2rgb(im_orig_tmp), 0, 1).astype(np.float32)


def histogram_equalize(im_orig):
    """
    Performs histogram equalization of a given grayscale or RGB image.
    If an RGB image is given, the following equalization procedure should only operate on the Y channel of
    the corresponding YIQ image and then convert back from YIQ to RGB. Moreover, the outputs hist_orig
    and hist_eq is the histogram of the Y channel only. The required intensity transformation is de-
    fined such that the gray levels should have an approximately uniform gray-level histogram (i.e. equalized
    histogram) stretched over the entire [0, 1] gray level range.
    :param im_orig: the input grayscale or RGB float32 image with values in [0, 1].
    :return: im_eq - is the equalized image. grayscale or RGB float32 image with values in [0, 1].
             hist_orig - is a 256 bin histogram of the original image (array with shape (256,) ).
             hist_eq - is a 256 bin histogram of the equalized image (array with shape (256,) ).
    """
    # take Y channel from RBG image, work with floats to prevent errors in histogram calculations that i saw.
    if im_orig.ndim == RGB:
        im_orig_work = rgb2yiq(im_orig)[:, :, 0]
    elif im_orig.ndim == GRAY:
        im_orig_work = im_orig
    else:
        raise "Invalid image format!"


    # calculate histogram
    hist_orig, bounds = np.histogram(im_orig_work, NUM_OF_PIXELS)

    # calculate cumulative histogram, then normalize and stretch according to formula
    hist_cum_norm = np.cumsum(hist_orig)
    first_non_zero = hist_cum_norm.min()
    hist_cum_norm = (((hist_cum_norm - first_non_zero) / (im_orig_work.size - first_non_zero))
                     * (NUM_OF_PIXELS - 1)).round().astype(np.uint8)

    # replace values in image according to hist_cum_norm
    im_eq = hist_cum_norm[(im_orig_work * (NUM_OF_PIXELS - 1)).round().astype(np.uint8)]

    # calculate equalized histogram
    hist_eq, bounds_eq = np.histogram(im_eq, NUM_OF_PIXELS)

    # return image from y channel to RGB and normalize to values float between 0 - 1
    if im_orig.ndim == RGB:
        im_eq = return_y_channel_to_rgb(im_eq, im_orig)
    else:
        im_eq = im_eq.astype(np.float32) / (NUM_OF_PIXELS - 1)

    return [im_eq, hist_orig, hist_eq]


def quantize_rgb(im_orig, n_quant, n_iter):
    """
    Perform RBG quantization of a given RBG image.
    :param im_orig: the input grayscale or RGB image to be quantized (float32 image with values in [0, 1]).
    :param n_quant: the number of intensities your output im_quant image should have.
    :param n_iter: the maximum number of iterations of the optimization procedure (may converge earlier).
    :return: im_quant - the quantize output image.
             error - is an array with shape (n_iter,) (or less) of the total intensities error for
             each iteration in the quantization procedure.
    """
    im_orig_work = np.copy(im_orig)
    im_r = (im_orig[:,:,0])
    im_g = (im_orig[:,:,1])
    im_b = (im_orig[:,:,2])

    im_orig_work[:, :, 0], error_r = quantize(im_r, n_quant, n_iter)
    im_orig_work[:, :, 1], error_g = quantize(im_g, n_quant, n_iter)
    im_orig_work[:, :, 2], error_b = quantize(im_b, n_quant, n_iter)

    max_error_len = max(len(error_r),len(error_b),len(error_b))

    error_r = np.pad(error_r, (0,max_error_len - len(error_r)), 'minimum')
    error_g = np.pad(error_g, (0,max_error_len - len(error_g)), 'minimum')
    error_b = np.pad(error_b, (0,max_error_len - len(error_b)), 'minimum')
    error = (error_r + error_b + error_g) / 3

    return [im_orig_work, error]


def quantize(im_orig, n_quant, n_iter):
    """
    Performs optimal quantization of a given grayscale or RGB image(only y channel).
    :param im_orig: the input grayscale or RGB image to be quantized (float32 image with values in [0, 1]).
    :param n_quant: the number of intensities your output im_quant image should have.
    :param n_iter: the maximum number of iterations of the optimization procedure (may converge earlier).
    :return: im_quant - the quantize output image.
             error - is an array with shape (n_iter,) (or less) of the total intensities error for
             each iteration in the quantization procedure.
    """
    # Take Y channel from RBG image, if GRAY use image as is - float32 between 0 to 1
    if im_orig.ndim == RGB:
        im_orig_work = rgb2yiq(im_orig)[:, :, 0]
    elif im_orig.ndim == GRAY:
        im_orig_work = im_orig
    else:
        raise "Invalid image format!"

    # Setting initial Z values division such that each segment will contain approximately the same number of pixels
    hist_orig = np.histogram((im_orig_work*(NUM_OF_PIXELS - 1)).astype(np.uint32), NUM_OF_PIXELS)[0]
    hist_cum = np.cumsum(hist_orig).astype(np.uint32)

    # first z value = 0, last = 255
    z_values = np.zeros((n_quant + 1), np.float32)
    z_values[n_quant] = NUM_OF_PIXELS - 1
    for i in range(1, n_quant):
        z_values[i] = np.where(hist_cum > (hist_cum[-1] * (i/n_quant)))[0][0]

    # Iteration section
    z_values_last_iter = np.copy(z_values).astype(np.float32)
    q_values = np.zeros(n_quant, np.float32)
    error = np.empty([0])
    for i in range(n_iter):
        # checking for stop condition - if no change in z values in last iteration stops
        loop_err = 0
        z_values = np.round(z_values).astype(np.uint32)
        if i != 0:
            if np.array_equal(z_values_last_iter, z_values):
                break
            else:
                z_values_last_iter = np.copy(z_values)

        # calculating q_values
        for j in range(n_quant):
            q_values[j] = np.dot(hist_orig[z_values[j]:z_values[j + 1] + 1]
                                 , np.arange(z_values[j], z_values[j + 1] + 1)) \
                          / np.sum(hist_orig[z_values[j]:z_values[j + 1] + 1])

            # calculating total error introduced by quantization in this iteration
            if j == 0:
                loop_err += np.dot(np.square(np.arange(z_values[j], z_values[j + 1] + 1) - q_values[j])
                                   , hist_orig[z_values[j]:z_values[j + 1] + 1])
                continue
            loop_err += np.dot(np.square(np.arange(z_values[j] + 1, z_values[j+1] + 1) - q_values[j])
                               ,hist_orig[z_values[j] + 1:z_values[j+1] + 1])
        # calculating z_values
        for k in range(1, n_quant):
            z_values[k] = np.round((q_values[k - 1] + q_values[k]) / 2).astype(np.uint32)

        error = np.append(error, loop_err)

    # rounding q values
    q_values = np.round(q_values)

    # quantizing image before finishing
    for i in range(n_quant):
        np.putmask(im_orig_work, (z_values[i] / (NUM_OF_PIXELS - 1) < im_orig_work)
                   & (im_orig_work <= (z_values[i + 1] / (NUM_OF_PIXELS - 1))), q_values[i])

    # returning to float32 between 0-1 for rgb2yiq to work correctly
    im_orig_work = (im_orig_work / (NUM_OF_PIXELS - 1)).astype(np.float32)

    # return image from y channel to RGB and to values float between 0 - 1
    if im_orig.ndim == RGB:
        im_orig_tmp = rgb2yiq(im_orig)
        im_orig_tmp[:, :, 0] = im_orig_work
        im_orig_work = yiq2rgb(im_orig_tmp)
    else:
        im_orig_work = im_orig_work

    return [im_orig_work, error]

