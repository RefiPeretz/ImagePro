import sol4_utils as util
import sol4_add as add
import numpy as np
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt

def calculate_matrix(im):
    der_x_im, der_y_im = util.conv_der(im)
    matrix_1 = util.blur_spatial(der_x_im * der_x_im, 3)
    matrix_2 = util.blur_spatial(der_x_im * der_y_im, 3)
    matrix_3 = util.blur_spatial(der_y_im * der_x_im, 3)
    matrix_4 = util.blur_spatial(der_y_im * der_y_im, 3)
    return matrix_1,matrix_2,matrix_3,matrix_4




def harris_corner_detector(im):
    matrix_1,matrix_2,matrix_3,matrix_4 = calculate_matrix(im)
    k = 0.04
    tmp1 = (matrix_1*matrix_4 - matrix_2*matrix_3)
    tmp2 = k*((matrix_1 + matrix_4)**2)
    R = tmp1 - tmp2
    res_r = add.non_maximum_suppression(R)
    non_zero_index = np.argwhere(res_r == 1)
    tmp1 = np.copy(non_zero_index[:,0])
    tmp2 = np.copy(non_zero_index[:, 1])
    non_zero_index[:,0] = tmp2
    non_zero_index[:,1] = tmp1
    return non_zero_index

def calculate_patch(cor,K,desc_rad):
    x,y = cor[0],cor[1]
    xs = np.meshgrid(np.arange(x - desc_rad, x + desc_rad+1), np.arange(x - desc_rad, x + desc_rad+1))[1].reshape(1,K*K)
    ys = np.meshgrid(np.arange(y - desc_rad, y + desc_rad+1), np.arange(y - desc_rad, y + desc_rad+1))[0].reshape(1,K*K)
    return  np.array([xs[0],ys[0]])

def sample_descriptor(im, pos, desc_rad):
    #TODO should we calculate it?
    reduce_factor = 2 ** -2
    K = 2*desc_rad + 1
    desc_list = []
    for cor in pos:
        res_patch = map_coordinates(im,calculate_patch(reduce_factor*cor,K,desc_rad),order=1).reshape(K,K)
        #TODO normlize check if done correct?
        res_patch = (res_patch - np.mean(res_patch)) / (np.linalg.norm(res_patch -  np.mean(res_patch)))
        desc_list.append(res_patch)
    return np.dstack(desc_list)

def find_featuers(pyr):
    pos = add.spread_out_corners(pyr[0],7,7,7)
    #TODO should we use 3?
    return sample_descriptor(pyr[2],pos,3)




im = util.read_image("external/oxford1.jpg",1)
find_featuers(util.build_gaussian_pyramid(im,3,3)[0])
# papo = add.spread_out_corners(im,7,7,7)
# #papo = harris_corner_detector(im)
# plt.imshow(im, cmap=plt.cm.gray)
# x = papo[:,0]
# y = papo[:,1]
# plt.scatter(x,y)
# plt.show()