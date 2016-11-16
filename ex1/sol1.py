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
    hist, bins = np.histogram(im_mod.flatten(), 256, [0,255])
    cdf = np.cumsum(hist)
    print(cdf)
    maxC = max(cdf)
    minC = min(cdf)
    print(maxC ,minC)
    cdf = (255 * (cdf - minC) / (maxC - minC))
    print(cdf)
    #cdf = cdf * 255 / im_mod.size

    eq_image = np.interp(im_mod,bins[:-1],cdf)
    try_hist = np.histogram(cdf[im_mod], 256, [0,256])[0]

    eq_image_hist , eq_bins = np.histogram(eq_image, 256,  [0,256])

    if (len(im_orig.shape) == 3):
        print('here')
        im_orig[:,:,0] = eq_image.astype(np.float32) / 255
        eq_image = np.clip(yiq2rgb(im_orig),0,1)
    else:
        eq_image = im_mod
    #plt.imshow(eq_image)
    #plt.show()
    plt.figure()
    plt.plot(eq_image_hist)
    plt.figure()
    plt.plot(try_hist, color='red')
    plt.show()

def quantize(im_orig, n_quant, n_iter):

    if(len(im_orig.shape) == 3):
        print('here')
        im_orig = rgb2yiq(im_orig)
        im_mod = im_orig[:,:,0]
    else:
        im_mod = im_orig
    im_mod*= 255
    im_mod = im_mod.astype(np.uint32)
    hist_orig = np.histogram(im_mod,256,[0,256])[0]
    hist_cumsum = np.cumsum(hist_orig)
    values_Z = np.zeros((n_quant + 1,), dtype=np.uint32)
    normlizer = hist_cumsum[-1]

    for i in range(1,n_quant):
        values_Z[i] = np.argwhere(hist_cumsum > normlizer * (i/n_quant))[0]

    values_Z[n_quant] = 256
    index_matrix = np.arange(0,256)
    values_Q = np.zeros((n_quant,), dtype=np.uint32)
    #split_image= np.split(hist_orig,values_Z)
    new_values_Z = np.copy(values_Z)
    error_hist_q = np.array([])
    for i in range(n_iter):
        print(i)
        calc_error = 0
        split_image = np.split(hist_orig, values_Z)[1:-1]
        split_image_weights = np.split(index_matrix,values_Z)[1:-1]
        for j in range(len(values_Q)):
            values_Q[j] = (np.sum(split_image[j]*split_image_weights[j]) / np.sum(split_image[j])).round().astype(np.uint32)
            calc_error += np.dot(hist_orig[values_Z[j]:values_Z[j+1]],np.square(np.arange(values_Z[j],values_Z[j+1])- values_Q[j]))

        for j in range(len(values_Q) - 1):
            new_values_Z[j+1] = ((values_Q[j]+values_Q[j+1]) / 2).astype(np.uint32)

        error_hist_q = np.append(error_hist_q,calc_error)
        if(np.array_equal(new_values_Z,values_Z)):
            break
        else:
            values_Z = np.copy(new_values_Z)

    for j in range(len(new_values_Z) - 1):
        np.putmask(im_mod, (im_mod >= new_values_Z[j]) & (im_mod <= new_values_Z[j+1]),values_Q[j])

    if (len(im_orig.shape) == 3):
        print('here')
        im_orig[:,:,0] = im_mod.astype(np.float32) / 255
        im_mod = np.clip(yiq2rgb(im_orig),0,1)

    #plt.imshow(im_mod, cmap = plt.cm.gray)
    #plt.imshow(im_mod)
    plt.plot(error_hist_q)
    plt.show()
    return normlized_image(im_mod)



def quantize_rgb(im_orig, n_quant, n_iter):
    im_orig[:,:,0 ] = quantize(im_orig[:,:,0],n_quant,n_iter)
    im_orig[:, :,1] = quantize(im_orig[:, :,1], n_quant, n_iter)
    im_orig[:, :, 2] = quantize(im_orig[:, :, 2], n_quant, n_iter)
    #plt.imshow(im_mod, cmap = plt.cm.gray)
    plt.imshow(im_orig)
    #plt.plot(error_hist_q)
    plt.show()








print('Start main')
im = read_image('LowContrast.jpg',1)
quantize(im,25,10)
#quantize_rgb(im,3,10)
#im2 = read_image('hist.jpg',1)
#histogram_equalize(im)

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










