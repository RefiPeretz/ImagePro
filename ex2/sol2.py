#TODO delete
import sol1 as im_func
import numpy as np
from scipy.signal import convolve2d, convolve
import matplotlib.pyplot as plt
from scipy.misc import imread as imread, imsave as imsave
from skimage.color import rgb2gray
from math import floor


def calc_matrix(size,exp_shape):
    N = size
    x = np.arange(N)
    u = np.arange(N).reshape(N, 1)
    return np.exp(-2j * (np.pi * u * x / N)) if exp_shape == 1 else np.exp(2j * (np.pi * u * x / N))

def DFT(signal):
    work_sig = np.copy(signal)
    w_matrix = calc_matrix(signal.shape[0],1)
    return np.dot(w_matrix, work_sig).astype(np.complex128)

def IDFT(signal):
    work_sig = np.copy(signal)
    N = signal.shape[0]
    w_matrix = calc_matrix(N,2)
    return (np.dot(w_matrix, work_sig)/N)


def DFT2(image):
    return np.dot(DFT(image), calc_matrix(image.shape[1],1)).astype(np.complex128)

def IDFT2(image):
    return np.dot(IDFT(image), calc_matrix(image.shape[1],2)) / image.shape[1]

def conv_der(im):
    derv_X = convolve2d(im, np.array([1, 0, -1]).reshape(1,3), mode="same")
    drev_Y = convolve2d(im, np.array([[1], [0], [-1]]).reshape(3,1), mode="same")
    return np.sqrt(np.power(derv_X, 2) + np.power(drev_Y, 2))

def fourier_der(im):
    #TODO check if we need check odd or even right now
    im_DFT = DFT2(im)
    N_F, M_F, N , M = int(im_DFT.shape[1]/2), int(im_DFT.shape[0]/2), im_DFT.shape[1] , im_DFT.shape[0]
    u_X = np.tile(np.concatenate((np.arange(0,N_F,1),np.arange(-N_F,0,1))).reshape(1,N), (M,1))
    u_Y = np.tile(np.concatenate((np.arange(0, M_F, 1), np.arange(-M_F, 0, 1))).reshape(M, 1), (1, N))

    derv_X_dft = u_X * im_DFT
    derv_Y_dft =  u_Y * im_DFT
    derv_X, derv_Y = IDFT2((derv_X_dft) * (2j * np.pi / M)), IDFT2((derv_Y_dft) * (2j * np.pi / N))
    return (np.sqrt(np.abs(derv_X) ** 2 + np.abs(derv_Y) ** 2))


def gaus_1d(kernel_size):
    gaus_kernel = np.array([1, 1])
    for i in range(kernel_size - 2):
        gaus_kernel = convolve(gaus_kernel, np.array([1, 1]), mode ='full')
    return gaus_kernel

def gaus_2d(kernel_size):
    d1_kernel = gaus_1d(kernel_size).reshape(1,kernel_size)
    return convolve2d(d1_kernel, d1_kernel.T, mode='full').astype(np.float32)

def blur_spatial(im, kernel_size):
    gaus_kerenel = gaus_2d(kernel_size)
    gaus_kerenel /= np.sum(gaus_kerenel)
    return convolve2d(im, gaus_kerenel, mode='same', boundary='wrap')

def blur_fourier(im, kernel_size):
    gaus_kerenel = gaus_2d(kernel_size)
    gaus_kerenel /= np.sum(gaus_kerenel)
    N,M = im.shape[0],im.shape[1]
    i, j = floor(N / 2) + 1, floor(M / 2) + 1
    gaus_kernel_pad = np.zeros(shape=(N, M))
    print(gaus_kerenel)
    low_x, top_x = i-floor(kernel_size/2), i + floor(kernel_size/2) + 1
    low_y, top_y = j-floor(kernel_size/2), j + floor(kernel_size/2) + 1
    # print(low_x, top_x)
    # print(low_y, top_y)
    gaus_kernel_pad[low_x:top_x, low_y:top_y] = gaus_kerenel
    # pad_width_x = int((im.shape[0] - gaus_kerenel.shape[0])/2)
    # pad_width_y = int((im.shape[1] - gaus_kerenel.shape[1])/2)
    # gaus_kerenel = np.pad(gaus_kerenel,((pad_width_x,pad_width_x+1),(pad_width_y,pad_width_y+1)),mode='constant', constant_values=0)

    gaus_kernel_pad = np.fft.fftshift(gaus_kernel_pad)
    im_DFT, gaus_kerenel_DFT = DFT2(im),  DFT2(gaus_kernel_pad)
    return IDFT2(im_DFT*gaus_kerenel_DFT).astype(np.float32)



# papo = np.random.random(32)
# papo = np.copy(papo).reshape(32,1)
#
#
# r1= DFT(papo)
# r2 = np.fft.fft2(papo)
# # r1= DFT(papo)
# # r3=DFT(papo2)
# # r2 = (np.fft.fft2(papo2))
#

# print(np.allclose(r1,r2))
#
# r1= IDFT(DFT(papo))
# r2 = np.fft.ifft2(np.fft.fft2(papo))
# # r1= DFT(papo)
# # r3=DFT(papo2)
# # r2 = (np.fft.fft2(papo2))
#
# print(np.allclose(r1,r2))
#
#
# papo = im_func.read_image('monkey.jpg',1)
# #papo = np.random.random(16).reshape(4,4)
# r1 = DFT2(np.copy(papo))
# r2 = np.fft.fft2(np.copy(papo))
# print(np.allclose(r1,r2))
#
# papo = im_func.read_image('monkey.jpg',1)
# r3 = IDFT2(DFT2(np.copy(papo)))
# r4 = np.fft.ifft2(np.fft.fft2(np.copy(papo)))
# # print(r3)
# # print('\n')
# # print(r4)
# print(np.allclose(r3,r4))
#
#
# papo = im_func.read_image('jerusalem.jpg',1)
# conv_res = conv_der(papo)
#
# plt.imshow(conv_res, cmap=plt.cm.gray)
# plt.show()
#
# papo = im_func.read_image('jerusalem.jpg',1)
#
# fourier_res = fourier_der(papo)
#
# plt.imshow(fourier_res, cmap=plt.cm.gray)
# plt.show()
#
# print(fourier_res)
# papo = im_func.read_image('jerusalem.jpg',1)
#
# plt.imshow(blur_spatial(papo, 5),cmap=plt.cm.gray)
# plt.show()
# papo2 = np.zeros(shape=(16,10))
# papo = np.array([[1,2,4,2,1],[1,2,5,2,1],[1,2,6,2,1],[1,2,7,2,1],[1,2,8,2,1]])
# print(papo.shape)
#
# i,j = floor(16/2) + 1, floor(10/2) + 1
# papo2[i-2:i+3, j-2:j+3] = papo
# print(papo2)
# print(papo2.shape)
# print(papo2[i,j])
# x = np.arange(8).reshape(2, 4)
# idx = (1, 3)
# print(np.insert(x, idx, 999, axis=1))

papo = im_func.read_image('jerusalem.jpg',1)
blur = blur_fourier(papo,5)
plt.imshow(blur,cmap=plt.cm.gray)
plt.show()

print('done')
