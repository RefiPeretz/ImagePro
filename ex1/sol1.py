import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread as imread, imsave as imsave
from skimage.color import rgb2gray





def normlized_image(image):
    if(image.dtype != np.float32):
        image = image.astype(np.float32)
    if(image.max() > 1):
        image /= 255

    return image


def is_rgb(im):
    #TODO verify if there is any better way
    if(len(im.shape) == 3):
        return True
    else:
        return False

def validate_representation(representation):

    if representation != 1 and representation != 2:
        raise Exception("Unkonwn representaion")


def read_image(fileame, representation):
    validate_representation(representation)

    im = imread(fileame)
    if representation == 1 and is_rgb(im):
        # We should convert from Grayscale to RGB
        im =rgb2gray(im)
        return im.astype(np.float32)

    return normlized_image(im)


def imdisplay(filename, representation):
    validate_representation(representation)

    im = read_image(filename,representation)
    plt.figure()
    if representation == 1:
        plt.imshow(im, cmap=plt.cm.gray)
    else:
        plt.imshow(im)

def rgb2yiq(imRGB):
    trans = np.array([[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]]).astype(np.float32)
    return imRGB.dot(trans.T)

def yiq2rgb(imYIQ):
    trans = np.array([[1.0, 0.956, 0.621], [1.0, -0.272, -0.647], [1.0, -1.106, 1.703]]).astype(np.float32)
    return imYIQ.dot(trans.T)




def histogram_equalize(im_orig):

    if(is_rgb(im_orig)):
        im_orig = rgb2yiq(im_orig)
        im_mod = im_orig[:,:,0]
    else:
        im_mod = im_orig

    im_mod = (255*im_mod).astype(np.uint8)
    hist, bins = np.histogram(im_mod.flatten(), 256, [0,255])
    hist_cumsum = np.cumsum(hist)
    maxC = max(hist_cumsum)
    minC = min(hist_cumsum)
    hist_cumsum = (255 * (hist_cumsum - minC) / (maxC - minC))
    #TODO decide between methods
    eq_image = np.interp(im_mod,bins[:-1],hist_cumsum)
    #try_hist = np.histogram(hist_cumsum[im_mod], 256, [0,256])[0]

    eq_image_hist = np.histogram(eq_image, 256,  [0,256])[0]

    if (is_rgb(im_orig)):
        im_orig[:,:,0] = normlized_image(eq_image)
        eq_image = np.clip(yiq2rgb(im_orig),0,1)
    else:
        eq_image = normlized_image(eq_image)

    return [eq_image,hist,eq_image_hist]


def quantize(im_orig, n_quant, n_iter):

    if(is_rgb(im_orig)):
        im_orig = rgb2yiq(im_orig)
        im_mod = im_orig[:,:,0]
    else:
        im_mod = im_orig
    #Normlize matrix
    im_mod = (255*im_mod).astype(np.uint32)
    hist_orig = np.histogram(im_mod,256,[0,256])[0]
    #Calculate cumsum for intital division
    hist_cumsum = np.cumsum(hist_orig)
    values_Z = np.zeros((n_quant + 1,), dtype=int)
    normlizer = hist_cumsum[-1]
    #calculate initial division
    for i in range(1,n_quant):
        values_Z[i] = np.argwhere(hist_cumsum > normlizer * (i/n_quant))[0]

    values_Z[n_quant] = 255
    values_Q = np.zeros((n_quant,), dtype=int)
    new_values_Z = np.copy(values_Z)
    error_hist_q = []

    for it in range(n_iter):
        curr_err = 0


        z_low = 0
        for i in range(n_quant):
            #Calculate Q base on Z
            values_Q[i] = (hist_orig[values_Z[i]:values_Z[i + 1] + 1].dot(np.arange(values_Z[i], values_Z[i + 1] + 1)) /
                        np.sum(hist_orig[values_Z[i]:values_Z[i + 1] + 1])).astype(np.uint32)

            z_top = values_Z[i + 1]
            # calc error:
            curr_err += hist_orig[z_low:z_top + 1].dot(np.square(np.arange(z_low, z_top + 1) - values_Q[i]))
            z_low = z_top + 1

        #Calculate new z base on Q
        for i in range(0,n_quant-1):
            new_values_Z[i+1] = ((values_Q[i] + values_Q[i + 1]) / 2).astype(np.uint32)


        error_hist_q.append(curr_err)
        if not np.array_equal(new_values_Z, values_Z):
            values_Z = np.copy(new_values_Z)
        else:
            break
    #Update matrix pixcel values base on new Q and borders
    for j in range(len(values_Z) - 1):
        np.putmask(im_mod, (im_mod >= values_Z[j]) & (im_mod <= values_Z[j+1]),values_Q[j])

    if (is_rgb(im_orig)):
        im_orig[:,:,0] = im_mod.astype(np.float32) / 255
        im_mod = np.clip(yiq2rgb(im_orig),0,1)

    return [normlized_image(im_mod),error_hist_q]



def quantize_rgb(im_orig, n_quant, n_iter):
    im_orig[:,:,0 ] = quantize(im_orig[:,:,0],n_quant,n_iter)
    im_orig[:, :,1] = quantize(im_orig[:, :,1], n_quant, n_iter)
    im_orig[:, :, 2] = quantize(im_orig[:, :, 2], n_quant, n_iter)
    #plt.imshow(im_mod, cmap = plt.cm.gray)
    plt.imshow(im_orig)
    #plt.plot(error_hist_q)
    plt.show()








print('Start main')
im = read_image('jerusalem.jpg',1)
im2 = read_image('jerusalem.jpg',2)
#im = imdisplay('jerusalem.jpg',1)
#quantize(im,2,10)
#quantize_rgb(im,3,10)

# eq_im, hist , eq_hist = histogram_equalize(im)
# eq_im2, hist2 , eq_hist2 = histogram_equalize(im2)
# plt.figure()
# plt.subplot(2,2,1)
# plt.imshow(eq_im,cmap = plt.cm.gray)
# plt.subplot(2,2,2)
# plt.imshow(eq_im2)
# plt.subplot(2,2,3)
# plt.plot(eq_hist)
# plt.subplot(2,2,4)
# plt.plot(eq_hist2)
# plt.show()

q,err = quantize(im,4,10)
q2,err2 = quantize(im2,4,10)

plt.figure()
plt.subplot(2,2,1)
plt.imshow(q,cmap = plt.cm.gray)
plt.subplot(2,2,2)
plt.imshow(q2)
plt.subplot(2,2,3)
plt.plot(err)
plt.subplot(2,2,4)
plt.plot(err2)
plt.show()




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










