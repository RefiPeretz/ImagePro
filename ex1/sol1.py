import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread as imread, imsave as imsave
from skimage.color import rgb2gray




def normlized_image(image):
    im_float = image.astype(np.float32)
    im_float /= 255
    return im_float



def read_image(fileame, representation):
    print('Start func')
    # TODO add input validation
    im = imread(fileame)
    if(representation == 1 and len(im.shape) == 3):
        # We should convert from Grayscale to RGB
        print('rgb2gray')
        im = rgb2gray(im)

    return normlized_image(im)



print('Start main')
im = read_image('jerusalem.jpg',2)
print(im[70, 60])
plt.imshow(im)

imsave('gray.jpg',im)










