#####################Import#############################
import numpy as np
import sol5_utils
from skimage.color import rgb2gray as rgb2gray
from scipy.misc import imread as imread, imsave as imsave
import scipy as scipy
from keras.layers import Input, Dense, Convolution2D,Activation,merge
from keras.models import Model
from keras.optimizers import Adam
from scipy import ndimage
from random import randint
import matplotlib.pyplot as plt

#####################global#############################
normalization = 255

####################function############################

def read_image(fileame, representation):
    """A function that reads a given image file and converts it into a given representation
        input: filename (a string), representation (int value of 1 or 2)
        output: image."""

    im = np.float32(imread(fileame))/normalization  # reading the picture
    if (representation == 1):# grayscale image
        im_g = rgb2gray(im)

        return (im_g).astype(np.float32);

    else: # RGB image
        return (im).astype(np.float32);


def load_dataset(filenames, batch_size, corruption_func, crop_size):
    """ function that generate pairs of image patches. each time picking a random image,
            applying a random corruption, and extracting a random patch
        input: filenames, batch_size, corruption_func, crop_size
        output: data_generator - Python’s generator object which outputs random tuples of the form
            (source_batch, target_batch)"""
    img_dic = {}
    # creating python generator

    while True:

        target_batch = np.ones((batch_size,1,int(crop_size[0]),int(crop_size[1])))
        source_batch = np.ones((batch_size,1,int(crop_size[0]),int(crop_size[1])))

        # select an image's file name randomly from filenames
        for i in range(batch_size):
            imFileNme = np.random.choice(filenames);

            # creating image using read_image
            if (imFileNme in img_dic):
                im = img_dic.get(imFileNme);

            else:
                im = read_image(imFileNme, 1);
                img_dic[imFileNme] = im;

            #corrupt selected image
            cor_im = corruption_func(im);

            #randomly choosing the location of a patch
            location_y = randint(0,im.shape[0]-int(crop_size[0]))

            location_x = randint(0, im.shape[1]-int(crop_size[1]))

            patch_im_original = im[location_y:location_y+int(crop_size[0]),location_x:location_x+int(crop_size[1])];
            patch_im_corrupt = cor_im[location_y:location_y+int(crop_size[0]),location_x:location_x+int(crop_size[1])];

            #subtract the value 0.5
            patch_im_original = patch_im_original - 0.5;
            patch_im_corrupt = patch_im_corrupt - 0.5;

            source_batch[i,:,:,:] = patch_im_corrupt;
            target_batch[i,:,:,:] = patch_im_original;

        yield (source_batch.astype(np.float32),target_batch.astype(np.float32));


def resblock(input_tensor, num_channels):
    """ function that creating a residual block
        input: input_tensor, num_channels
        output: output_tensor"""

    conv1 = Convolution2D(num_channels, 3, 3, border_mode='same')(input_tensor);
    activ_relu = Activation ('relu')(conv1);
    conv2 = Convolution2D(num_channels, 3, 3, border_mode='same')(activ_relu);

    outPut_tensor = merge([input_tensor, conv2 ] ,mode ='sum')

    return outPut_tensor;


def build_nn_model(height, width, num_channels):
    """ function which returns the complete neural network model
        input: height, width, num_channels
        output: model"""

    input_l = Input(shape =(1, height , width));

    conv1 = Convolution2D(num_channels, 3, 3, border_mode='same')(input_l);

    activ_relu = Activation('relu')(conv1);
    tempBlock = activ_relu
    for i in range(5):
        tempBlock = resblock(tempBlock,num_channels);

    temp_tensor = merge([activ_relu, tempBlock ] ,mode ='sum');
    last_conv = Convolution2D(1, 3, 3, border_mode='same')(temp_tensor);

    model = Model(input = input_l , output = last_conv );
    return model;

def train_model(model, images, corruption_func, batch_size,
                                    samples_per_epoch, num_epochs, num_valid_samples):
    """ function that train the model
     input: model, images, corruption_func, batch_size,samples_per_epoch, num_epochs, num_valid_samples
     output:"""

    #creating list of train and test
    listOfImg_train = images[:int(len(images)*(0.8))]
    listOfImg_test = images[int(len(images) -int(len(images) * (0.2))):]

    #creating database of training and testing
    db_trian = load_dataset(listOfImg_train, batch_size, corruption_func, model.input_shape[2:4]);
    db_test = load_dataset(listOfImg_test, batch_size, corruption_func, model.input_shape[2:4]);

    #compilation
    model.compile(loss='mean_squared_error',optimizer= Adam(beta_2 = 0.9));

    #training the model
    model.fit_generator(db_trian,samples_per_epoch=samples_per_epoch, nb_epoch=num_epochs,
                        validation_data=db_test, nb_val_samples=num_valid_samples);



def restore_image(corrupted_image, base_model, num_channels):
    """ restore full images of any size
        input: corrupted_image, base_model, num_channels
        output: restored_image"""
    corrupted_image_new_model = np.zeros(shape=(1,corrupted_image.shape[0],corrupted_image.shape[1]))
    corrupted_image_new_model[0, :, :] = corrupted_image

    #building new model for given picture
    pic_model = build_nn_model(corrupted_image.shape[0], corrupted_image.shape[1], num_channels);

    #copy weights from the given model to the adjusted model
    pic_model.set_weights(base_model.get_weights());

    #restoring the image
    corrupted_image_new_model[0, :, :] =  corrupted_image_new_model[0, :, :] - 0.5  # todo - do we need that?
    corrupted_image_new_model = pic_model.predict(corrupted_image_new_model[np.newaxis ,...])[0]
    temp_pic = corrupted_image_new_model[0, :, :] + 0.5

    #cliping
    temp_pic = np.clip(temp_pic, 0, 1);
    return temp_pic;

def add_gaussian_noise(image, min_sigma, max_sigma):
    """ random noise function for training
        input: image, min_sigma, max_sigma
        output: corrupted"""

    sigma = np.random.uniform(min_sigma,max_sigma);
    gaussian_noise = np.random.normal(0,sigma,image.shape);
    image = image + gaussian_noise;
    return (np.clip(image,0,1)).astype(np.float32)


def learn_denoising_model(quick_mode=False):
    """ random noise function for training
        input: quick_mode=False
        output: model, num_channels """

    list_to_learn = sol5_utils.images_for_denoising();
    num_channels = 48;
    min_sigma = 0;
    max_sigma = 0.2
    height = 24;
    width = 24;

    #inizilaize
    if(quick_mode == False):

        batch_size = 100;
        samples_per_epoch = 10000;
        num_epochs = 5;
        num_valid_samples = 1000;

    if(quick_mode == True):
        batch_size = 10;
        samples_per_epoch = 30;
        num_epochs = 2;
        num_valid_samples = 30;

    #building the model
    denoising_model = build_nn_model(height,width,num_channels)

    # training the model
    train_model(denoising_model,list_to_learn,\
                lambda pic: add_gaussian_noise(pic, min_sigma, max_sigma),\
                batch_size,samples_per_epoch,num_epochs,num_valid_samples);

    return denoising_model,num_channels;


def add_motion_blur(image, kernel_size, angle):
    """ simulate motion blur on the given image using a square kernel
        of size kernel_size where the line (as described above) has
        the given angle in radians
        input: image, kernel_size, angle
        output: corrupted """

    kernal = sol5_utils.motion_blur_kernel(kernel_size, angle);
    image = scipy.ndimage.filters.convolve(image,kernal);
    return image;


def random_motion_blur(image, list_of_kernel_sizes):
    """ samples an angle at uniform from the range
        [0, π), and choses a kernel size at uniform from the list
        list_of_kernel_sizes, followed by applying the previous function with the given
        image and the randomly sampled parameters
        input: image, list_of_kernel_sizes
        output: corrupted """

    return add_motion_blur(image, np.random.choice(list_of_kernel_sizes), np.random.uniform(0,np.pi))


def learn_deblurring_model(quick_mode=False):
    """ return a trained deblurring model, and the number of channels used in its construction
        input: quick_mode=False
        output: model, num_channels """

    list_to_learn = sol5_utils.images_for_deblurring();
    num_channels = 32;
    height = 16;
    width = 16;

    #inizilaize
    if (quick_mode == False):
        batch_size = 100;
        samples_per_epoch = 10000;
        num_epochs = 10;
        num_valid_samples = 1000;

    if (quick_mode == True):
        batch_size = 10;
        samples_per_epoch = 30;
        num_epochs = 2;
        num_valid_samples = 30;

    # building the model
    debloring_model = build_nn_model(height, width, num_channels)

    # training the model
    train_model(debloring_model, list_to_learn, \
                lambda pic: random_motion_blur(pic, [7]), \
                batch_size, samples_per_epoch, num_epochs, num_valid_samples);

    return debloring_model, num_channels;


#################################  test ######################################


#filenames = [f for f in listdir('/cs/usr/rotembh/ex5/text_dataset/train') if isfile(join('/cs/usr/rotembh/ex5/text_dataset/train', f))]
# im1 = read_image('/cs/usr/rotembh/ex5/text_dataset/train/'+filenames[20],1);
# im2 = random_motion_blur(im1, [7])
# model = build_nn_model(16, 16, 32)
# model.load_weights('debloring_model.h5')
# re_im = restore_image(im2,model,32);
# fig = plt.figure("pictures:");
# sub = fig.add_subplot(221,title='im')
# plt.imshow(im1,cmap=plt.cm.gray)
# sub = fig.add_subplot(222, title='im2')
# plt.imshow(im2,cmap=plt.cm.gray)
# sub = fig.add_subplot(223, title='re_im')
# plt.imshow(re_im,cmap=plt.cm.gray)
# plt.show()



# filenames2 = [f for f in listdir('/cs/usr/rotembh/ex5/image_dataset/train') if isfile(join('/cs/usr/rotembh/ex5/image_dataset/train', f))]
# im11 = read_image('/cs/usr/rotembh/ex5/image_dataset/train/'+filenames2[5],1);
# im2 = add_gaussian_noise(im11, 0, 0.2)
# model = build_nn_model(24, 24, 48)
# model.load_weights('denois_model.h5')
# re_im = restore_image(im2,model,48);
# fig = plt.figure("pictures:");
# sub = fig.add_subplot(221,title='im')
# plt.imshow(im11,cmap=plt.cm.gray)
# sub = fig.add_subplot(222, title='im2')
# plt.imshow(im2,cmap=plt.cm.gray)
# sub = fig.add_subplot(223, title='re_im')
# plt.imshow(re_im,cmap=plt.cm.gray)
# plt.show()
# corruption_func = lambda x: x
# crop_size = (16, 16)
# image_sets = [sol5_utils.images_for_denoising(), sol5_utils.images_for_deblurring()]
# train_set = load_dataset(image_sets[0], 100, corruption_func, crop_size);
#
# source, target =next(train_set)
# print(source.dtype)
# source, target =next(train_set)
# print(source.dtype)
# source, target =next(train_set)
# print(source.dtype)
