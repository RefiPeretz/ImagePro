import numpy as np
from scipy.misc import imread as imread, imsave as imsave




def read_image(fileame, representation):
    # TODO add input validation
    im = imread(fileame)
