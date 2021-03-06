import sol3
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc

MAX_PIXEL_VALUE = 255
IMG_GRAYSCALE = 1
IMG_RGB = 2

rgb_im = 'externals_example/road_examp.jpg'

pow_of_two_img = np.zeros(2**10, dtype=np.float32)


# def check_float32(var, name):
#     string = name
#     if var.dtype != np.float32:
#         string += " ***Error*** Not float32"
#     else:
#         string += " OK - dtype is float32"
#     print(string)

def check_between_zero_to_one(var, name):
    string = name
    if var.max() > 1:
        string += " ***Error*** Not between [0,1]"
    else:
        string += " OK - range is between [0,1]"
    print(string)


rgb_to_gray = sol3.read_image(rgb_im, IMG_GRAYSCALE)

plt.figure("rgb_to_gray")
plt.imshow(rgb_to_gray, cmap=plt.cm.gray)

g_pyr, filter_vec = sol3.build_gaussian_pyramid(rgb_to_gray, 10, 3)

fig = plt.figure("GAUSSIAN PYRAMID")
for i in range(len(g_pyr)):
    fig.add_subplot(1, len(g_pyr), i + 1)
    plt.imshow(g_pyr[i], cmap=plt.cm.gray)


for i in range(len(g_pyr)):
    #check_float32(g_pyr[i], "G[" + str(i) + "]")
    check_between_zero_to_one(g_pyr[i], "G[" + str(i) + "]")

l_pyr, l_filter_vec = sol3.build_laplacian_pyramid(rgb_to_gray, 10, 3)

fig = plt.figure("LAPLACIAN PYRAMID")
for i in range(len(l_pyr)):
    fig.add_subplot(1, len(l_pyr), i + 1)
    plt.imshow(l_pyr[i], cmap=plt.cm.gray)

reconstructed = sol3.laplacian_to_image(l_pyr, filter_vec, np.array([1, 1, 1, 1, 1, 1, 1]))
#check_float32(reconstructed, "reconstructed")
check_between_zero_to_one(reconstructed, "reconstructed")

fig = plt.figure("RECONSTRUCTED")
plt.imshow(reconstructed, cmap=plt.cm.gray)

print(np.all(abs(rgb_to_gray - reconstructed) < 1e-12))
print(np.max(abs(rgb_to_gray - reconstructed)))

rendered_lpyr = sol3.render_pyramid(l_pyr, 15)
#check_float32(rendered_lpyr, "reconstructed")
check_between_zero_to_one(rendered_lpyr, "reconstructed")

rendered_gpyr = sol3.render_pyramid(g_pyr, 15)

#check_float32(rendered_gpyr, "reconstructed")
check_between_zero_to_one(rendered_gpyr, "reconstructed")

sol3.display_pyramid(l_pyr, 15)
sol3.display_pyramid(g_pyr, 15)

# im1 = sol3.read_image('external/owl.jpg', IMG_GRAYSCALE)
# im2 = sol3.read_image('external/lena_1024x1024_rgb.jpg', IMG_GRAYSCALE)
# mask = sol3.read_image('external/mask.jpg', IMG_GRAYSCALE)
#
# blended = sol3.pyramid_blending(im1, im2, mask, 4, 3, 3)
# fig = plt.figure("BLENDED")
# plt.imshow(blended, cmap=plt.cm.gray)
#
mask1 =  sol3.blending_example1()[2]
plt.show()
mask2 = sol3.blending_example2()[2]
plt.show()

# mask = sol3.read_image(sol3.relpath('externals_example/road_mask_im.jpg'), IMG_GRAYSCALE)
# hist ,bins = np.histogram(mask, bins=3)
# mask = sol3.read_image(sol3.relpath('externals_example/eye_mask_im.jpg'), IMG_GRAYSCALE)
# sol3.blending_example1()

for i in range(mask2.shape[0]):
    for j in range(mask2.shape[1]):
        if mask1[i][j] > 0 and mask1[i][j] < 1:
            print('fuck')
        if mask2[i][j] > 0 and mask2[i][j] < 1:
            print('fuck')

# plt.show()
