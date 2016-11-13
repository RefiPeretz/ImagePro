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
def imdisplay(filename, representation):
    plt.figure()
    plt.imshow(read_image(filename,representation))


def papo(im):
    #TODO input validation
    saveRows = im.shape[0]
    saveCols = im.shape[1]
    im = im.reshape(saveRows*saveCols,3)
    im = im.T
    #trans = np.array([[0.299,0.596,0.212],[0.587,-0.275,-0.523],[0.114,-0.321,0.311]])
    trans = np.array([[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]])
    im = np.dot(trans,im)
    im = im.T
    return im.reshape(saveRows,saveCols,3)

def papo1(im):
    #TODO input validation
    saveRows = im.shape[0]
    saveCols = im.shape[1]
    im = im.reshape(saveRows*saveCols,3)
    im = im.T
    trans = np.array([[1.0,0.956,0.621],[1.0,-0.272,-0.647],[1.0,-1.106,1.703]])
    #trans = np.array([[1.0, 1.0, 1.0], [0.956, -0.272, -1.106], [0.621, -0.647, 1.703]])
    im = np.dot(trans,im)
    im = im.T
    return im.reshape(saveRows,saveCols,3)







print('Start main')
im = read_image('jerusalem.jpg',2)
#imdisplay('jerusalem.jpg',2)
#im = im[300:304,200:204,:]
im = papo(im)
#im = papo1(im)


imsave('gray.jpg',im)










