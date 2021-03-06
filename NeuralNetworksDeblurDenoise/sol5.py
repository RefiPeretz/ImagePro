import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread as imread, imsave as imsave
from scipy import misc, signal, ndimage
from skimage.color import rgb2gray
import random
import sol5_utils as util
from keras.layers import Input, Convolution2D, Dense, Activation, merge
from keras.models import Model
import math
from keras.optimizers import Adam

DENOISING_HEIGHT, DENOISING_WIDTH = 24,24
DENOISING_NUM_CHANNEL = 48
GAUS_NOISE_MIN = 0
GAUS_NOISE_MAX = 0.2
DEBLURING_HEIGHT, DEBLURING_WIDTH = 16, 16
DEBLURING_NUM_CHANNEL = 32
LIST_MOTION_BLUR = [7]

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
    """
    a generator that returns a dataset with clean and corrupted images

    :param filenames: A list of file name of the images
    :param batch_size: The size of the batch of images
    :param corruption_func: A corruption function.
    :param crop_size: indicate the size of the patches
    :return: (source_batch, target_batch) - sets of courpted and clean images.
    """
    ims_dic = {}
    num_of_files = len(filenames)
    height, width = int(crop_size[0]), int(crop_size[1])
    while True:
        target_batch = np.zeros(shape=(batch_size,1,height,width)).astype(np.float32)
        source_batch = np.copy(target_batch)

        for i in range(batch_size):

            rand_int = random.randint(0,num_of_files-1)
            cur_file_name = filenames[rand_int]

            if cur_file_name in ims_dic:
                work_im = ims_dic[cur_file_name]
            else:
                work_im = read_image(util.relpath(cur_file_name), 1)
                ims_dic[cur_file_name] = np.copy(work_im)

            c_work_im = corruption_func(np.copy(work_im))

            x, y = random.randint(0, work_im.shape[0] - height), \
                   random.randint(0, work_im.shape[1] - width)

            path_orig, patch_cor = (work_im[x:x + height, y:y + width] - 0.5).astype(np.float32), \
                                   (c_work_im[x:x + height, y:y + width] - 0.5).astype(np.float32)

            target_batch[i, :, :, :], source_batch[i, :, :, :] = path_orig, patch_cor

        yield (source_batch, target_batch)


def resblock(input_tensor, num_channels):
    """
    Build a resdual block as explained in class.

    :param input_tensor: a symbolic input tensor
    :param num_channels: number of channels for the convolution layer.
    :return: output_tensor: the output tensor for the residual block
    """

    conv = Convolution2D(num_channels, 3, 3, border_mode='same')(input_tensor)
    relu = Activation('relu')(conv)
    conv = Convolution2D(num_channels, 3, 3, border_mode='same')(relu)
    return merge([input_tensor, conv], mode='sum')


def build_nn_model(height, width, num_channels):
    """
    Generate a resNet model

    :param height: height
    :param width: width
    :param number of channels for the convolution layer.
    :return: ResNet model
    """
    input_tensor = Input(shape=(1, height, width))
    #Bulding the begining of the nn.
    input_start = Convolution2D(num_channels, 3, 3, border_mode='same')(input_tensor)
    input_start = Activation('relu')(input_start)
    tensors_chain = input_start

    #Build residual blocks.
    for i in range(5):
        tensors_chain = resblock(tensors_chain,num_channels)

    output = merge([tensors_chain, input_start], mode='sum')

    output = Convolution2D(1, 3, 3, border_mode='same')(output)

    return Model(input=input_tensor, output=output)


def train_model(model, images, corruption_func, batch_size, samples_per_epoch,
                num_epochs, num_valid_samples):
    """
    trains the nueral networks

    :param model: The nn model
    :param images: Set of images we train on/
    :param corruption_func: A corruption func we want to learn.
    :param batch_size: Size of batch we working on for each iteration.
    :param samples_per_epoch:
    :param num_epochs:
    :param num_valid_samples: Size of validation test.
    """
    split_by = math.floor(len(images)*0.8)
    training_set = images[:split_by]
    validation_set = images[split_by:]

    training_gen = load_dataset(training_set, batch_size, corruption_func,
                                 (model.input_shape[2], model.input_shape[3]))
    validation_gen = load_dataset(validation_set, batch_size, corruption_func,
                                 (model.input_shape[2], model.input_shape[3]))

    model.compile(loss='mean_squared_error', optimizer=Adam(beta_2=0.9))
    model.fit_generator(training_gen, samples_per_epoch=samples_per_epoch,
                        nb_epoch=num_epochs,
                        validation_data=validation_gen,
                        nb_val_samples=num_valid_samples)


def restore_image(corrupted_image, base_model, num_channels):
    """

    :param corrupted_image:
            An image which got corrupted by a certain function.
    :param base_model:
            A NN model we build desin to solve the problem.
    :param number of channels for the convolution layer.
    :return: restored_image
    """
    height, width = int(corrupted_image.shape[0]), int(corrupted_image.shape[1])

    res_model = build_nn_model(height, width, num_channels)

    res_model.set_weights(base_model.get_weights())

    corrupted_image_reshape = np.zeros(shape=(1, 1,height, width))
    corrupted_image_reshape[0, :, :, :] = corrupted_image - 0.5

    predicted = res_model.predict(corrupted_image_reshape)[0, 0]
    return np.clip(predicted + 0.5, 0, 1)


def add_gaussian_noise(image, min_sigma, max_sigma):
    """
    :param image:
    :param min_sigma:
    :param max_sigma:
    :return: image after gaussian noise was added to it.
    """
    height, width = int(image.shape[0]), int(image.shape[1])
    add_simga = random.uniform(min_sigma, max_sigma)
    image += np.random.normal(scale=add_simga,size=(height, width))
    return np.clip(image, 0, 1)


def learn_denoising_model(quick_mode=False):
    """
    :param quick_mode:
    :return:denoising_model - Trained nn, DENOISING_NUM_CHANNEL - number of channel we used.
    """

    lam_cor = lambda im: add_gaussian_noise(im, GAUS_NOISE_MIN, GAUS_NOISE_MAX)
    ims = util.images_for_denoising()
    ##Build model for desnosing
    denoising_model = build_nn_model(DENOISING_HEIGHT,DENOISING_WIDTH,DENOISING_NUM_CHANNEL)
    ##TODO change qucikmode to orig
    train_model(denoising_model, ims, lam_cor, 10, 30, 2, 30) if quick_mode else \
        train_model(denoising_model, ims, lam_cor, 100, 10000, 5, 1000)

    return denoising_model, DENOISING_NUM_CHANNEL


def add_motion_blur(image, kernel_size, angle):
    """
    :param image: an image to blur
    :param kernel_size: size for the blur kernel
    :param angle: an angle between zero to PI
    :return: blurred image
    """

    return ndimage.filters.convolve(image, util.motion_blur_kernel(kernel_size, angle))


def random_motion_blur(image, list_of_kernel_sizes):
    """
    :param image: an image to blur
    :param list_of_kernel_sizes:
    :return: randomly blurred image
    """
    rand_angle = random.uniform(0,np.pi)
    rand_int = random.randint(0, len(list_of_kernel_sizes) - 1)
    kernel = list_of_kernel_sizes[rand_int]
    return add_motion_blur(image, kernel, rand_angle)


def learn_deblurring_model(quick_mode=False):
    """
    :param quick_mode:
    :return:debluring_model - Trained nn, DEBLURING_NUM_CHANNEL - number of channel we used.
    """

    lam_cor = lambda im: random_motion_blur(im, LIST_MOTION_BLUR)
    ims = util.images_for_deblurring()
    debluring_model = build_nn_model(DEBLURING_HEIGHT, DEBLURING_WIDTH, DEBLURING_NUM_CHANNEL)
    train_model(debluring_model, ims, lam_cor,  10, 30, 2, 30) if quick_mode else \
        train_model(debluring_model, ims, lam_cor,  100, 10000, 10, 1000)

    return debluring_model, DEBLURING_NUM_CHANNEL
