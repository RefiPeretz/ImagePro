import sol1 as im_func
import numpy as np
from scipy.signal import convolve2d, convolve
import matplotlib.pyplot as plt
from scipy.misc import imread as imread, imsave as imsave
from skimage.color import rgb2gray


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
    # N = np.round(im.shape[1] / 2)
    #
    # im_dx = np.fft.fftshift(DFT2(im)) * np.arange(-N, N, 1)
    # im_dx = IDFT2(np.fft.ifftshift(im_dx)) * (2 * np.pi * 1j / N)
    # # y derivative
    # M = np.round(im.shape[0] / 2)
    # im_dy = (np.fft.fftshift(DFT2(im)).T * np.arange(-M, M, 1)).T
    # im_dy = IDFT2(np.fft.ifftshift(im_dy)) * (2 * np.pi * 1j / M)  # TODO devide again by M or N?
    # return (np.sqrt(np.abs(im_dx) * 2 + np.abs(im_dy) * 2)).astype(np.float32)
    im_DFT = np.fft.fftshift(DFT2(im))
    N_F, M_F, N , M = np.round(im_DFT.shape[1]/2), np.round(im_DFT.shape[0]/2), im_DFT.shape[1] , im_DFT.shape[0]
    derv_X_dft = im_DFT*np.arange(-N_F, N_F,1)
    derv_Y_dft = im_DFT.T * np.arange(-M_F, M_F,1).T
    derv_X, derv_Y = DFT2(np.fft.ifftshift(derv_X_dft)) * (2j * np.pi / N), DFT2(np.fft.ifftshift(derv_Y_dft)) * (2j * np.pi / N)
    return (np.sqrt(np.abs(derv_X) * 2 + np.abs(derv_Y) * 2)).astype(np.float32)


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
# papo = np.random.random(16).reshape(4,4)
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

papo = im_func.read_image('jerusalem.jpg',1)

fourier_res = fourier_der(papo)

plt.imshow(fourier_res, cmap=plt.cm.gray)
plt.show()

print('done')
