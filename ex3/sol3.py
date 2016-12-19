import sol2
import numpy as np
from scipy.signal import convolve2d, convolve
from scipy import ndimage
from scipy.misc import imread as imread
from skimage.color import rgb2gray
from math import floor
import matplotlib.pyplot as plt

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

def expand_im(im,filter_vec):
    rowSize , colSize = im.shape[0],im.shape[1]
    #zero pad image
    exp_img = np.zeros((2*rowSize, 2*colSize), dtype=np.float32)
    exp_img[::2, ::2] = im[:, :]
    exp_img = ndimage.filters.convolve(exp_img, 2 * filter_vec.T)
    return ndimage.filters.convolve(exp_img, 2 * filter_vec)


def build_laplacian_pyramid(im, max_levels, filter_size):
    filter_vec = gaus_1d(filter_size).reshape(1,filter_size)
    g_pyr = build_gaussian_pyramid(im,max_levels,filter_size)[0]
    l_pyr = []
    for i in range(len(g_pyr) - 1):
        print(i)
        l_im = g_pyr[i] - expand_im(g_pyr[i+1],filter_vec)
        minIm , maxIm = l_im.min(), l_im.max()
        l_im = (l_im - minIm) / (maxIm - minIm)
        l_pyr.append(l_im)


    plt.imshow(g_pyr[-1], cmap=plt.cm.gray)
    plt.show()
    l_pyr.append(g_pyr[-1])
    return [l_pyr,filter_vec]






def build_gaussian_pyramid(im, max_levels, filter_size):
    #TODO do we need to append original image
    filter_vec = gaus_1d(filter_size).reshape(1,filter_size)
    pyr = []
    pyr.append(im)
    for i in range(max_levels):
        #print(im.shape)
        #TODO do we need 'same'?
        #TODO do we need to multiply by 2?
        im = ndimage.filters.convolve(im, 2*filter_vec.T)
        im = ndimage.filters.convolve(im, 2*filter_vec)
        #TODO this is ok method?
        im = im[::2, ::2]
        minIm , maxIm = im.min(), im.max()
        im = (im - minIm) / (maxIm - minIm)
        pyr.append(im)
    #TODO (1,filter_size) or (,filter_size)
    return [pyr,filter_vec]

def render_pyramid(pyr, levels):
    colRes = 0
    for i in range(levels):
        colRes += pyr[i].shape[1]
    rowRes = pyr[0].shape[0]
    resIm = np.zeros((rowRes,colRes),dtype=np.float32)
    print(resIm.shape)
    print(pyr[0].shape)
    curCol, curRow = 0,0
    for i in range(levels):
        resIm[curRow : pyr[i].shape[0],curCol:pyr[i].shape[1] + curCol] = pyr[i]
        curCol += pyr[i].shape[1]

    plt.imshow(resIm, cmap=plt.cm.gray)
    plt.show()


# papo = sol2.read_image('test.jpg',1)
#
# papo , vec = build_gaussian_pyramid(papo,5,3)
#
# # for i in range(5):
# #     print(papo[i].shape)
# #     plt.imshow(papo[i], cmap=plt.cm.gray)
# #     plt.show()
# render_pyramid(papo,5)

papo = sol2.read_image('test.jpg',1)

papo , vec = build_laplacian_pyramid(papo,3,3)

render_pyramid(papo,3)


# def iexpand(image):
#   out = None
#   kernel = generating_kernel(0.4)
#   outimage = np.zeros((image.shape[0]*2, image.shape[1]*2), dtype=np.float64)
#   outimage[::2,::2]=image[:,:]
#   out = 4*scipy.signal.convolve2d(outimage,kernel,'same')
#   return out