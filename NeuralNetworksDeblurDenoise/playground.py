import sol5_utils as util
import sol5
import matplotlib.pyplot as plt

#quick = False
quick = False

################# DENOISING TEST #################

im_path = util.images_for_denoising()[22]

model = sol5.learn_denoising_model(quick)[0]
model.save_weights("denoising_weights_full.weights")

im = sol5.read_image(im_path, 1)

cor_im = sol5.add_gaussian_noise(sol5.read_image(im_path,1),0,0.2)

res_im = sol5.restore_image(cor_im, model, 48)

ax1 = plt.subplot(221)
ax1.set_title("cor_im")
plt.imshow(cor_im, cmap='gray')

ax2 = plt.subplot(222)
ax2.set_title("restored Image")
plt.imshow(res_im, cmap='gray')

ax3 = plt.subplot(223)
ax3.set_title("original")
plt.imshow(im, cmap='gray')


plt.show()


################ DEBLURRING TEST #################


im_path = util.images_for_deblurring()[23]


model = sol5.learn_deblurring_model(quick)[0]
model.save_weights("deblurring_weights_full.weights")


im = sol5.read_image(im_path, 1)

cor_im = sol5.random_motion_blur(sol5.read_image(im_path,1), [7])

res_im = sol5.restore_image(cor_im, model, 32)

ax1 = plt.subplot(221)
ax1.set_title("cor_im")
plt.imshow(cor_im, cmap='gray')

ax2 = plt.subplot(222)
ax2.set_title("restored Image")
plt.imshow(res_im, cmap='gray')

ax3 = plt.subplot(223)
ax3.set_title("original")
plt.imshow(im, cmap='gray')


plt.show()