import sol1
import matplotlib.pyplot as plt

#file = 'jerusalem.jpg'
#file = 'LowContrast.jpg'
file = 'monkey.jpg'

fig = plt.figure()
#
sub = fig.add_subplot(221, title='Original')
im_org = sol1.read_image(file, 2)
imgplot = plt.imshow(im_org)
#
sub = fig.add_subplot(222, title='Original')
im_org_gray = sol1.read_image(file, 1)
imgplot = plt.imshow(im_org_gray, cmap=plt.cm.gray)
#
sub = fig.add_subplot(223, title='rgb2yiq(img)')
yiq_img = sol1.rgb2yiq(im_org)
imgplot = plt.imshow(yiq_img)
#
sub = fig.add_subplot(224, title='yiq2rgb(yiq_img)')
yiq2rgb = sol1.yiq2rgb(yiq_img)
imgplot = plt.imshow(yiq2rgb)
#
# # check histogram_equalize for grayscale
fig = plt.figure()
im_eq , hist_orig , hist_eq = sol1.histogram_equalize(im_org_gray)
#
sub = fig.add_subplot(321, title='im_org_gray')
plt.imshow(im_org_gray, cmap=plt.cm.gray)
#
sub = fig.add_subplot(322, title='im_org_gray_eq')
plt.imshow(im_eq, cmap=plt.cm.gray)
#
sub = fig.add_subplot(323, title='hist_orig')
plt.plot(hist_orig)
#
sub = fig.add_subplot(324, title='hist_eq')
plt.plot(hist_eq)
#
#sub = fig.add_subplot(325, title='hist_orig_eq')
#plt.plot(sol1.normalize_hist(hist_orig))
# #
#sub = fig.add_subplot(326, title='hist_eq_eq')
#plt.plot(sol1.normalize_hist(hist_eq))
#
# # check histogram_equalize for rgb
fig = plt.figure()
im_eq , hist_orig , hist_eq = sol1.histogram_equalize(im_org)
#
sub = fig.add_subplot(321, title='im_org')
plt.imshow(im_org, cmap=plt.cm.gray)
#
sub = fig.add_subplot(322, title='im_org_eq')
plt.imshow(im_eq, cmap=plt.cm.gray)
#
sub = fig.add_subplot(323, title='hist_orig')
plt.plot(hist_orig)
#
sub = fig.add_subplot(324, title='hist_eq')
plt.plot(hist_eq)
#
#sub = fig.add_subplot(325, title='hist_orig_eq')
#plt.plot(sol1.normalize_hist(hist_orig))
# #
#sub = fig.add_subplot(326, title='hist_eq_eq')
#plt.plot(sol1.normalize_hist(hist_eq))
#

# check quantize for grayscale
fig = plt.figure()
im_quant, error = sol1.quantize(im_org_gray, 20, 50)

sub = fig.add_subplot(321, title='im_org_gray')
plt.imshow(im_org_gray, cmap=plt.cm.gray)

sub = fig.add_subplot(322, title='im_org_gray_quant')
plt.imshow(im_quant, cmap=plt.cm.gray)

sub = fig.add_subplot(323, title='error')
plt.plot(error)

# check quantize for rgb
fig = plt.figure()
im_quant1, error = sol1.quantize(im_org, 20, 50)

sub = fig.add_subplot(321, title='im_org')
plt.imshow(im_org)

sub = fig.add_subplot(322, title='im_org_quant')
plt.imshow(im_quant1)

sub = fig.add_subplot(323, title='error')
plt.plot(error)

#check quantize for rgb real
fig = plt.figure()
im_quant1 = sol1.quantize_rgb(sol1.np.copy(im_org), 4, 50)

sub = fig.add_subplot(321, title='im_org')
plt.imshow(im_org)

sub = fig.add_subplot(322, title='im_org_quant_rgb')
plt.imshow(im_quant1[0])

# sub = fig.add_subplot(325, title='hist_orig_eq')
# plt.plot(sol1.normalize_hist(hist_orig))
#
# sub = fig.add_subplot(326, title='hist_eq_eq')
# plt.plot(sol1.normalize_hist(hist_eq))

plt.show()
