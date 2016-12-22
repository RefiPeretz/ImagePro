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
    exp_img = np.zeros((2*rowSize, 2*colSize))
    exp_img[::2, ::2] = im
    exp_img = ndimage.filters.convolve(exp_img, 2 * filter_vec.T)
    return ndimage.filters.convolve(exp_img, 2 * filter_vec)


def build_laplacian_pyramid(im, max_levels, filter_size):
    filter_vec = gaus_1d(filter_size).reshape(1, filter_size)
    g_pyr = build_gaussian_pyramid(im, max_levels, filter_size)[0]
    l_pyr = []
    for i in range(len(g_pyr) - 1):
        l_im = g_pyr[i] - expand_im(g_pyr[i + 1], filter_vec)
        l_pyr.append(l_im)

    l_pyr.append(g_pyr[-1])
    return [l_pyr, filter_vec]






def build_gaussian_pyramid(im, max_levels, filter_size):
    """

    """

    #TODO do we need to append original image
    filter_vec = gaus_1d(filter_size).reshape(1,filter_size)
    pyr = []
    pyr.append(im)
    for i in range(max_levels - 1):
        #print(im.shape)
        #TODO do we need 'same'?
        #TODO do we need to multiply by 2?
        im = ndimage.filters.convolve(im, filter_vec.T)
        im = ndimage.filters.convolve(im, filter_vec)
        #TODO this is ok method?
        im = im[::2, ::2]
        pyr.append(im)
    #TODO (1,filter_size) or (,filter_size)
    return [pyr,filter_vec]



def render_pyramid(pyr, levels):
    colRes = 0
    for i in range(levels):
        colRes += pyr[i].shape[1]
    rowRes = pyr[0].shape[0]
    resIm = np.zeros((rowRes,colRes),dtype=np.float32)
    curCol, curRow = 0,0
    for i in range(levels):
        minIm , maxIm = np.min(pyr[i]), np.max(pyr[i])
        pyr[i] = (pyr[i] - minIm) / (maxIm - minIm)
        resIm[curRow : pyr[i].shape[0],curCol:pyr[i].shape[1] + curCol] = pyr[i]
        curCol += pyr[i].shape[1]

    return resIm



def display_pyramid(pyr, levels):
    plt.imshow(render_pyramid(pyr,levels), cmap=plt.cm.gray)
    plt.show()


def laplacian_to_image(lpyr, filter_vec, coeff):
    #TODO check size
    size_list = len(coeff)
    for i in range(size_list):
        lpyr[i] *= coeff[i]
    print(filter_vec)

    resIm = lpyr[size_list-1]
    for i in range(size_list- 1,0,-1):
        print(i)
        resIm = expand_im(resIm,filter_vec)
        resIm += lpyr[i-1]

    return resIm



def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    l_im1_pyr,filter_vec = build_laplacian_pyramid(im1,max_levels,filter_size_im)
    l_im2_pyr = build_laplacian_pyramid(im2,max_levels,filter_size_im)[0]
    g_mask_pyr = build_gaussian_pyramid(mask.astype(np.float32),max_levels,filter_size_mask)[0]
    #TODO this is the len?
    l_out = []
    for i in range(len(l_im1_pyr)):
        tmp1 = (g_mask_pyr[i]*l_im1_pyr[i])
        tmp2 = (1 - g_mask_pyr[i])*l_im2_pyr[i]
        l_out.append(tmp1 + tmp2)
    #TODO this is the len?
    return np.clip(laplacian_to_image(l_out,filter_vec,[1]*len(l_im1_pyr)),0,1)


def check_line_right(x,y):
    if((y-x*1.9) > -675):
        return True
    return False

def check_line_left(x,y):
    if((y+x*1.88) < 1680):
        return True
    return False



def blending_example1():
    #TODO blend need to be bool?
    blend1 = sol2.read_image('sea.jpg', 2)
    blend2 = sol2.read_image('road.jpg', 2)
    plt.imshow(blend2, cmap=plt.cm.gray)
    plt.show()





    print(blend2.shape)
    print(blend1.shape)
    mask = np.zeros((1024, 1024))
    mask = mask.astype(np.bool)
    for i in range(639,890):
        for j in range(1024):
            mask[i,j] = check_line_right(i,j)
            if(mask[i,j] == 0.0):
                mask[i, j] = check_line_left(i, j)



    print(mask[849:880,:3])



    mask[592:639,:505] = True
    mask[586:639,518:] = True


    plt.imshow(mask, cmap=plt.cm.gray)
    plt.show()


    #mask[825:,:] = False
    mask_res = np.zeros((1024,1024,3))

    mask_res[:,:,0] = pyramid_blending(blend1[:,:,0], blend2[:,:,0], mask, 5, 21,7)
    mask_res[:,:,1] = pyramid_blending(blend1[:,:,1], blend2[:,:,1], mask, 5, 21, 7)
    mask_res[:,:,2] = pyramid_blending(blend1[:,:,2], blend2[:,:,2], mask, 5, 21, 7)


    return mask_res





# papo = sol2.read_image('test.jpg',1)
#
# papo , vec = build_gaussian_pyramid(papo,5,3)
#
# # for i in range(5):
# #     print(papo[i].shape)
# #     plt.imshow(papo[i], cmap=plt.cm.gray)
# #     plt.show()
# render_pyramid(papo,5)

papo_im = sol2.read_image('test.jpg',1)
#
# papo , vec = build_laplacian_pyramid(papo_im,4,3)
#
# #display_pyramid(papo,4)
# papo_im2 = laplacian_to_image(papo,vec,[1,1,1,1])
#
# plt.imshow(papo_im2,cmap=plt.cm.gray)
# plt.show()
#
# plt.imshow(papo_im,cmap=plt.cm.gray)
# plt.show()
#
# print(np.all(abs(papo_im - papo_im2) < 10**-12))

# def iexpand(image):
#   out = None
#   kernel = generating_kernel(0.4)
#   outimage = np.zeros((image.shape[0]*2, image.shape[1]*2), dtype=np.float64)
#   outimage[::2,::2]=image[:,:]
#   out = 4*scipy.signal.convolve2d(outimage,kernel,'same')
#   return out


plt.imshow(blending_example1(), cmap=plt.cm.gray)
plt.show()


