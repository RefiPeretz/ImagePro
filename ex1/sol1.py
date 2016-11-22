import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread as imread, imsave as imsave
from skimage.color import rgb2gray





def normlized_image(image):
    """
    Normlize image to float 32 and [0,1]
    :param image: Reprentaion of display 1 for grayscale 2 for RGB
    :return normlized image
    """
    if(image.dtype != np.float32):
        image = image.astype(np.float32)
    if(image.max() > 1):
        image /= 255

    return image


def is_rgb(im):
    """
    Verify if an image is RGB
    :param im: Reprentaion of display 1 for grayscale 2 for RGB
    """
    if(im.ndim == 3):
        return True
    else:
        return False


def validate_representation(representation):
    """
    Validate reprentaion input
    :param representation: Reprentaion of display 1 for grayscale 2 for RGB
    """

    if representation != 1 and representation != 2:
        raise Exception("Unkonwn representaion")


def read_image(fileame, representation):
    """
    Read image by file name and normlize it to float 32 [0,1] representaion
    according to RGB or Graysacle
    :param filename: The name of the file that we should read.
    :param representation: Reprentaion of display 1 for grayscale 2 for RGB
    :return normlized image
    """
    validate_representation(representation)

    im = imread(fileame)
    if representation == 1 and is_rgb(im):
        # We should convert from Grayscale to RGB
        im =rgb2gray(im)
        return im.astype(np.float32)

    return normlized_image(im)


def imdisplay(filename, representation):
    """
    Read and display image
    :param filename: The name of the file that we should read.
    :param representation: Reprentaion of display 1 for grayscale 2 for RGB
    """
    validate_representation(representation)

    im = read_image(filename,representation)
    plt.figure()
    if representation == 1:
        plt.imshow(im, cmap=plt.cm.gray)
    else:
        plt.imshow(im)

def rgb2yiq(imRGB):
    """
    Transform an float 32 [0,1] RGB image to float32 [0,1]  YIQ image
    :param im_orig: Original image
    :return: YIQ format image
    """
    trans = np.array([[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]]).astype(np.float32)
    return imRGB.dot(trans.T)

def yiq2rgb(imYIQ):
    """
    Transform an float 32 [0,1] YIQ image to float32 [0,1] RGB image
    :param im_orig: Original image
    :return: RGB format image
    """
    trans = np.array([[1.0, 0.956, 0.621], [1.0, -0.272, -0.647], [1.0, -1.106, 1.703]]).astype(np.float32)
    return imYIQ.dot(trans.T)




def histogram_equalize(im_orig):
    """
    Function that performs histogram equalization of a given grayscale or RGB image
    :param im_orig: Original image
    :return: Equalize image, original histogram, equalize image's histogram
    """

    if(is_rgb(im_orig)):
        im_orig = rgb2yiq(im_orig)
        im_mod = im_orig[:,:,0]
    else:
        im_mod = im_orig

    im_mod = (255*im_mod).astype(np.uint8)
    #TODO do we need [0,255]
    hist, bins = np.histogram(im_mod.flatten(), 256, [0,255])
    hist_cumsum = np.cumsum(hist)
    maxC = max(hist_cumsum)
    minC = min(hist_cumsum)
    hist_cumsum = (255 * (hist_cumsum - minC) / (maxC - minC))
    #TODO decide between methods
    eq_image = hist_cumsum[im_mod]


    eq_image_hist = np.histogram(eq_image, 256,  [0,256])[0]

    if (is_rgb(im_orig)):
        im_orig[:,:,0] = normlized_image(eq_image)
        eq_image = np.clip(yiq2rgb(im_orig),0,1)
    else:
        eq_image = normlized_image(eq_image)

    return [eq_image,hist,eq_image_hist]


def quantize(im_orig, n_quant, n_iter):
    """
    Quantize image: function that performs optimal quantization of a given grayscale or RGB image
    grayscale.
    :param im_orig: Original image
    :param n_quant: How many quants
    :param n_iter: number of max allowed iterations
    :return: Quantiz image and error graph
    """

    if(is_rgb(im_orig)):
        im_orig = rgb2yiq(im_orig)
        im_mod = im_orig[:,:,0]
    else:
        im_mod = im_orig
    #Normlize matrix
    im_mod = (255*im_mod).astype(int)
    hist_orig = np.histogram(im_mod,256,[0,256])[0]
    #Calculate cumsum for intital division
    hist_cumsum = np.cumsum(hist_orig)
    hist_orig = hist_orig.astype(np.float32)
    print(hist_cumsum)
    values_Z = np.zeros((n_quant + 1,), dtype=np.float32)
    normlizer = hist_cumsum[-1]
    #calculate initial division
    for i in range(1,n_quant):
        values_Z[i] = np.argwhere(hist_cumsum > normlizer * (i/n_quant))[0]

    values_Z[n_quant] = 255.0
    values_Q = np.zeros((n_quant,), dtype=np.float32)
    new_values_Z = np.copy(values_Z)
    error_hist_q = []

    for it in range(n_iter):
        print(it)
        curr_err = 0

        for i in range(n_quant):
            #Calculate Q base on Z
            cur_low_border = values_Z[i].astype(np.uint32)
            cur_top_border = values_Z[i+1].astype(np.uint32) + 1
            temp1 = (hist_orig[cur_low_border:cur_top_border].dot(np.arange(cur_low_border, cur_top_border))).astype(np.float32)
            temp2 = np.sum(hist_orig[cur_low_border:cur_top_border])
            values_Q[i] = temp1/temp2

            # # calc error:
            curr_err += hist_orig[cur_low_border:cur_top_border].dot(np.square(np.arange(cur_low_border, cur_top_border) - values_Q[i]))



        #Calculate new z base on Q
        for i in range(0,n_quant-1):
            temp = (values_Q[i] + values_Q[i + 1]).astype(np.float32)
            temp /= 2
            new_values_Z[i+1] = temp


        error_hist_q.append(curr_err)
        new_values_Z = np.round(new_values_Z)

        if not np.array_equal(new_values_Z, values_Z):
            values_Z = np.copy(new_values_Z)
        else:
            break
    #Update matrix pixcel values base on new Q and borders
    #values_Q = np.round(values_Q)
    for j in range(n_quant):
        # if(values_Z[j] == values_Z[j+1]):
        #     values_Z[j+1] = values_Z[j+1] + 1
        # if(j != n_quant-1 and values_Q[j] == values_Q[j+1]):
        #     values_Q[j+1] = values_Q[j+1] + 1
        if(j == 0):
            np.putmask(im_mod, (im_mod >= values_Z[j]) & (im_mod <= values_Z[j + 1]), values_Q[j].round())
            continue

        np.putmask(im_mod, (im_mod > values_Z[j]) & (im_mod <= values_Z[j+1]) ,values_Q[j].round())
    # #TODO delete
    papo = np.unique(np.copy(im_mod))
    for i in range(len(papo)):
        print(papo[i],values_Q[i], i, values_Z[i],values_Z[i+1])

    if (is_rgb(im_orig)):
        im_orig[:,:,0] = im_mod.astype(np.float32) / 255
        im_mod = np.clip(yiq2rgb(im_orig),0,1)

    return [normlized_image(im_mod),np.array(error_hist_q,)]

def pad_list(l,max_size,pad_with):
    """
    Pads list to a given size with given value.
    :param l: List to pad
    :param max_size: Pad list to max_size
    :param pad_with: pad list with this value
    greyscale image (1) or an RGB image (2)
    :return: Paded list.
    """
    return l + [pad_with]*(max_size - len(l))

def quantize_rgb(im_orig, n_quant, n_iter):
    """
    Bonus mission quantize full rgb by quantize every channel of RGB in separate.
    Then shape it back to image. We use the quantize method refering the channels as
    grayscale.
    :param im_orig: Original image
    :param n_quant: How many quants
    :param n_iter: number of max allowed iterations
    greyscale image (1) or an RGB image (2)
    :return: Quantize RGB image and error grahph
    """
    im_work = np.copy(im_orig)
    # Divide quantizie to 3 channels
    im_work[:,:,0 ],err_red = quantize(im_work[:,:,0],n_quant,n_iter)
    im_work[:, :,1],err_green = quantize(im_work[:, :,1], n_quant, n_iter)
    im_work[:, :, 2],err_blue = quantize(im_work[:, :, 2], n_quant, n_iter)
    err_red, err_green, err_blue = err_red.tolist(), err_green.tolist(), err_blue.tolist()
    # Calculate error by padding the lists of errors according to the maximum list
    max_list = max(len(err_red),len(err_green), len(err_blue))
    # Pad error list with last error value of each list to the max size.
    err_red,err_green,err_blue = pad_list(err_red,max_list,err_red[-1]),pad_list(err_green,max_list,err_green[-1]),pad_list(err_blue,max_list,err_blue[-1])
    calc_error = np.array([x + y + z for x, y ,z in zip(err_red, err_green,err_blue)]).astype(np.float32)
    calc_error /= 3
    return [im_work,calc_error]








print('Start main')
jr = 'jerusalem.jpg'
ben = 'LowContrast.jpg'
mon = 'monkey.jpg'

im = read_image(mon,1)
im2 = read_image(jr,2)
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
qn =70
it = 40

q,err = quantize(im,qn,it)
#q2,err2 = quantize(im2,qn,it)
q = (255*q).astype(np.uint8)
papo = np.histogram(q,256,[0,256])[0]
print(np.count_nonzero(papo))
plt.plot(err)
plt.show()

#
#
# plt.figure()
#
# plt.subplot(2,2,1)
# plt.imshow(q,cmap = plt.cm.gray)
# plt.subplot(2,2,2)
# plt.imshow(q2)
# plt.subplot(2,2,3)
# plt.plot(err)
# plt.subplot(2,2,4)
# plt.plot(err2)
# plt.title('quant = ' + str(qn) + 'iteration = ' +str(it))
# plt.show()


# quantize_rgb(im2,qn,it)

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

