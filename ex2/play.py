import sol2
import numpy as np
import matplotlib.pyplot as plt
# import sol2.py from non local folder
import sol2

def check_float32(var, name):
    print("Checking ", name, " dtype...")
    if var.dtype != np.float32:
        print("**Error** Not float32")
    else:
        print("dtype is OK")

fname = 'monkey.jpg'

############ TEST 1.1 #############
x = np.random.rand(1024,1).astype(np.float32)
print("1.1 - checking DFT:")
print(np.allclose(sol2.DFT(x), np.fft.fft2(x)))
print("1.1 - checking IDFT:")
print(np.allclose(sol2.IDFT(sol2.DFT(x)), np.fft.ifft2(np.fft.fft2(x))))

############ TEST 1.2 #############
x = np.random.rand(1024,512).astype(np.float32)
print("1.2 - checking 2D DFT:")
print(np.allclose(sol2.DFT2(x), np.fft.fft2(x)))
print("1.2 - checking 2D IDFT:")
print(np.allclose(sol2.IDFT2(sol2.DFT2(x)), np.fft.ifft2(np.fft.fft2(x))))

############ TEST 2.1 #############
im_org_gray = sol2.read_image(fname, 1)
magnitude = sol2.conv_der(sol2.read_image(fname,1))


check_float32(magnitude, "conv_der")

fig = plt.figure("TEST 2.1")

sub = fig.add_subplot(131, title='im_org_gray')
plt.imshow(im_org_gray, cmap=plt.cm.gray)
sub = fig.add_subplot(132, title='im_org_gray_cov_der')
plt.imshow(magnitude, cmap=plt.cm.gray)

############ TEST 2.2 #############
im_org_gray = sol2.read_image(fname, 1)
magnitude1 = sol2.fourier_der(sol2.read_image(fname,1))
check_float32(magnitude1, "fourier_der")

fig = plt.figure("TEST 2.2")
sub = fig.add_subplot(121, title='im_org_gray')
plt.imshow(im_org_gray, cmap=plt.cm.gray)
sub = fig.add_subplot(122, title='im_org_gray_fourier_magnitude')
plt.imshow(magnitude1, cmap=plt.cm.gray)

############ TEST 3.1 #############
im_org_gray = sol2.read_image(fname, 1)
fig = plt.figure("TEST 3.1")
sub = fig.add_subplot(121, title='im_org_gray')
plt.imshow(im_org_gray, cmap=plt.cm.gray)

sub = fig.add_subplot(122, title='im_org_gray_blurring_in_image_space')
blur0 = sol2.blur_spatial(im_org_gray,9)
check_float32(blur0, "blur_spatial")
plt.imshow(blur0, cmap=plt.cm.gray)


############ TEST 3.2 #############
im_org_gray = sol2.read_image(fname, 1)
blur1 = sol2.blur_fourier(im_org_gray,9)
check_float32(blur1, "blur_fourier")
fig = plt.figure("TEST 3.2")
sub = fig.add_subplot(121, title='im_org_gray')
plt.imshow(im_org_gray, cmap=plt.cm.gray)

sub = fig.add_subplot(122, title='im_org_gray_blurring_in_fourier_space')
plt.imshow(blur1, cmap=plt.cm.gray)

print("3.2/3.1 - comparing blurring:")
print(np.allclose(blur0, blur1))

plt.show()

x = np.random.rand(7,7).astype(np.float32)
sol2.blur_fourier(x, 3)