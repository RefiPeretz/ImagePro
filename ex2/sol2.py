import sol1 as im_func
import numpy as np
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
    # r1 , r2 = IDFT(image), np.fft.ifft2(np.copy(image))
    # print(r1)
    # print('\n')
    # print(r2)
    # print(np.allclose(r1,r2))
    return np.dot(IDFT(image), calc_matrix(image.shape[1],2)) / image.shape[1]

papo = np.random.random(32)
papo = np.copy(papo).reshape(32,1)


r1= IDFT(DFT(papo))
r2 = np.fft.ifft(np.fft.fft(papo)).real.astype(np.float32)
# r1= DFT(papo)
# r3=DFT(papo2)
# r2 = (np.fft.fft2(papo2))

print(np.allclose(r1,r2))


# im = im_func.read_image('monkey.jpg',1)
papo = np.random.random(16).reshape(4,4)
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

print('done')
