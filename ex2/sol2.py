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
    im_DFT = np.fft.fftshift(DFT2(im))
    derv_X_dft = np.fft.fftshift((im_DFT * (np.arange(im_DFT.shape[0]).reshape(im_DFT.shape[0], 1))))
    derv_Y_dft = np.fft.fftshift((im_DFT * (np.arange(im_DFT.shape[1]).reshape(im_DFT.shape[1], 1))))
    return np.sqrt(np.abs(derv_X_dft) ** 2 + np.abs(derv_Y_dft) ** 2)


papo = np.random.random(32)
papo = np.copy(papo).reshape(32,1)




r1= IDFT(DFT(papo))
r2 = np.fft.ifft(np.fft.fft(papo))
# r1= DFT(papo)
# r3=DFT(papo2)
# r2 = (np.fft.fft2(papo2))

print(np.allclose(r1,r2))


papo = im_func.read_image('monkey.jpg',1)
#papo = np.random.random(16).reshape(4,4)
r1 = DFT2(np.copy(papo))
r2 = np.fft.fft2(np.copy(papo))
print(np.allclose(r1,r2))

papo = np.random.random(16).reshape(4,4)
r3 = IDFT2(DFT2(np.copy(papo)))
r4 = np.fft.ifft2(np.fft.fft2(np.copy(papo)))
# print(r3)
# print('\n')
# print(r4)
print(np.allclose(r3,r4))


papo = im_func.read_image('jerusalem.jpg',1)
conv_res = conv_der(papo)

plt.imshow(conv_res, cmap=plt.cm.gray)
plt.show()

papo = im_func.read_image('jerusalem.jpg',1)

fourier_res = fourier_der(papo)

plt.imshow(conv_res, cmap=plt.cm.gray)
plt.show()

print('done')
