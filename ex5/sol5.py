import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread as imread, imsave as imsave
from skimage.color import rgb2gray
import random
import sol5_utils as util
from keras.layers import Input, Convolution2D, Dense, Activation, merge
from keras.models import Model
from keras.optimizers import Adam



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
    num_of_files = len(filenames)
    highet, width = int(crop_size[0]), int(crop_size[1])
    while True:
        target_batch = np.zeros((batch_size,1,int(crop_size[0]),int(crop_size[1])), dtype=np.folat32)
        source_batch = np.copy(target_batch)

        for i in range(batch_size):

            rand_int = random.randint(0,num_of_files)
            cur_file_name = filenames[rand_int]
            work_im = None

            if cur_file_name in ims_dic:
                work_im = ims_dic[cur_file_name]
            else:
                work_im = read_image(util.relpath(cur_file_name), 1)
                ims_dic[cur_file_name] = np.copy(work_im)

            c_work_im = corruption_func(np.copy(work_im))

            x, y = random.randint(0, work_im.shape[0] - highet), \
                   random.randint(0, work_im.shape[1] - width)

            path_orig, patch_cor = (work_im[x:x + highet, y:y + width] - 0.5).astype(np.float32), \
                                   (c_work_im[x:x + highet, y:y + width] - 0.5).astype(np.float32)

            target_batch[i, :, :, :], source_batch[i, :, :, :] = path_orig, patch_cor

            yield source_batch,target_batch


def resblock(input_tensor, num_channels):
    conv = Convolution2D(num_channels, 3, 3, border_mode='same')(input_tensor)
    relu = Activation('relu')(conv)
    conv = Convolution2D(num_channels, 3, 3, border_mode='same')(relu)
    return merge([input_tensor, conv], mode='sum')


def build_nn_model(height, width, num_channels):
    input_tensor = Input(shape=(1, height, width))
    #Bulding the begining of the nn.
    input_start = Convolution2D(num_channels, 3, 3, border_mode='same')(input_tensor)
    input_start = Activation('relu')(input_start)
    tensors_chain = input_start

    #TODO we want 5?
    #Build residual blocks.
    for i in range(5):
        tensors_chain = resblock(tensors_chain,num_channels)

    output = merge([tensors_chain, input_start], mode='sum')

    output = Convolution2D(1, 3, 3, border_mode='same')(output)

    return Model(input=input_tensor, output=output)








