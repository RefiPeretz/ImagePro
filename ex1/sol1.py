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
        im =rgb2gray(im)
        im = im.astype(np.float32)
        return im

    return normlized_image(im)
def imdisplay(filename, representation):
    im = read_image(filename,representation)
    if(representation == 1):
        plt.imshow(im, cmap=plt.cm.gray)
    else:
        plt.imshow(im)
    plt.show()



def rgb2yiq(imRGB):
    trans = np.array([[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]]).astype(np.float32)
    return imRGB.dot(trans.T)

def yiq2rgb(imYIQ):
    trans = np.array([[1.0, 0.956, 0.621], [1.0, -0.272, -0.647], [1.0, -1.106, 1.703]]).astype(np.float32)
    return imYIQ.dot(trans.T)
#
# def histogram_equalize(im_orig):
#     im_orig.astype(np.float32)
#     #hist = np.histogram(im_orig, bins=255)
#     im_orig*=255
#     hist_orig, bounds = np.histogram(im_orig.flatten(), bins=256)
#     print(hist_orig)
#     #plt.plot((bounds[:-1] + bounds[1:]) / 2, hist)
#     hist_cumsum = np.cumsum(hist_orig).astype(np.float32)
#     # Normlize
#
#     hist_cumsum = 255*hist_cumsum / hist_cumsum[-1]
#    # hist_cumsum = hist_cumsum.round().astype(np.uint8)
#     print(hist_cumsum)
#
#     #Equalize image
#     #im_eq = hist_cumsum[(im_orig.flatten().round()).astype(np.uint8)]
#     im_eq = np.interp(im_orig.flatten(), bounds[:-1], hist_cumsum)
#     im_eq = im_eq.reshape(im_orig.shape)
#     im_eq = normlized_image(im_eq)
#     #Extract new histogram
#     hist_eq, bounds_eq =  np.histogram(im_eq, bins=256)
#     plt.plot(hist_eq)
#     plt.show()
#     plt.imshow(im_eq, cmap=plt.cm.gray)
#     plt.show()
#
#     return [im_eq, hist_orig, hist_eq]
#
#     # #extract new histogram



def histogram_equalize(im_orig):

    if(len(im_orig.shape) == 3):
        print('here')
        im_orig = rgb2yiq(im_orig)
        im_mod = im_orig[:,:,0]
    else:
        im_mod = im_orig

    im_mod *=255
    im_mod = im_mod.astype(np.uint8)
    hist, bins = np.histogram(im_mod.flatten(), 256, [0,256])
    cdf = np.cumsum(hist)
    print(cdf)
    maxC = max(cdf)
    minC = min(cdf)
    print(maxC ,minC)
    cdf = (255 * (cdf - minC) / (maxC - minC))
    print(cdf)
    #cdf = cdf * 255 / im_mod.size

    eq_image = np.interp(im_mod,bins[:-1],cdf)

    eq_image_hist , eq_bins = np.histogram(eq_image, 256,  [0,256])

    if (len(im_orig.shape) == 3):
        print('here')
        im_orig[:,:,0] = eq_image.astype(np.float32) / 255
        eq_image = np.clip(yiq2rgb(im_orig),0,1)
    else:
        eq_image = im_mod
    plt.imshow(eq_image)
    plt.show()
    plt.plot(eq_image_hist)
    plt.show()



print('Start main')
im = read_image('LowContrast.jpg',2)
im2 = read_image('hist.jpg',1)
histogram_equalize(im)

#imdisplay('jerusalem.jpg',1)
#im = im[300:304,200:204,:]
#im1 = papo(im)
#print(im1[70][60])
#im3 = read_image('jerusalem.jpg',2)
#im3 = papo2(im3)
#print(im3[70][60])
#im3 = papo3(im3)
#print(im3[70][60])

#imsave('hist.jpg',im)










