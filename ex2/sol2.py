import sol1 as im_func
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread as imread, imsave as imsave
from skimage.color import rgb2gray


def DFT(signal):
    work_sig = np.copy(signal)
    N = signal.shape[0]
    x = np.arange(N)
    u = np.arange(N).reshape((N, 1))
    complex_vec = np.exp(-2j * (np.pi * u * x / N))
    return np.dot(complex_vec, work_sig).astype(np.complex128)





papo = np.random.random(32)
DFT(papo.reshape(32,1))