import sol4
import sol4_od
import sol4_utils
import sol4_add
import numpy as np
import matplotlib.pyplot as plt

REPRESENTATION_GRAYSCALE = 1
REPRESENTATION_RGB = 2

Img_1 = sol4_utils.read_image("external/backyard1.jpg", REPRESENTATION_GRAYSCALE)
Img_2 = sol4_utils.read_image("external/backyard2.jpg", REPRESENTATION_GRAYSCALE)
Img_3 = sol4_utils.read_image("external/backyard3.jpg", REPRESENTATION_GRAYSCALE)

### TEST 3.1 ###
# Img_1_feature_points = sol4_add.spread_out_corners(Img_1, 7, 7, 7)
# Img_1_feature_points_x = [x[0] for x in Img_1_feature_points]
# Img_1_feature_points_y = [y[1] for y in Img_1_feature_points]
# #Img_1_feature_points = sol4.harris_corner_detector(Img_1)
# Img_2_feature_points = sol4_add.spread_out_corners(Img_2, 7, 7, 7)
# Img_2_feature_points_x = [x[0] for x in Img_2_feature_points]
# Img_2_feature_points_y = [y[1] for y in Img_2_feature_points]
# #Img_2_feature_points = sol4.harris_corner_detector(Img_2)
#
# fig = plt.figure("TEST 3.1 - Harris")
# sub = fig.add_subplot(121, title='Im_1')
# plt.imshow(Img_1, cmap=plt.cm.gray)
# plt.scatter(Img_1_feature_points_x, Img_1_feature_points_y)
# sub = fig.add_subplot(122, title='Im_2')
# plt.imshow(Img_2, cmap=plt.cm.gray)
# plt.scatter(Img_2_feature_points_x, Img_2_feature_points_y)

### TEST 3.2 ###
Img_1_descriptor = sol4.find_features(sol4_utils.build_gaussian_pyramid(Img_1, 3, 7)[0]) #TODO filter_size?
Img_2_descriptor = sol4.find_features(sol4_utils.build_gaussian_pyramid(Img_2, 3, 7)[0]) #TODO filter_size?
Img_3_descriptor = sol4.find_features(sol4_utils.build_gaussian_pyramid(Img_3, 3, 7)[0]) #TODO filter_size?

min_score = 0.4
match_ind1, match_ind2 = sol4.match_features(Img_1_descriptor[1], Img_2_descriptor[1], min_score)
Img_1_matched_indexes = Img_1_descriptor[0][match_ind1]
Img_1_feature_points_x = [x[0] for x in Img_1_matched_indexes]
Img_1_feature_points_y = [y[1] for y in Img_1_matched_indexes]
Img_2_matched_indexes = Img_2_descriptor[0][match_ind2]
Img_2_feature_points_x = [x[0] for x in Img_2_matched_indexes]
Img_2_feature_points_y = [y[1] for y in Img_2_matched_indexes]
match_ind2_2, match_ind3_2 = sol4.match_features(Img_2_descriptor[1], Img_3_descriptor[1], min_score)
Img_2_2_matched_indexes = Img_2_descriptor[0][match_ind2_2]
Img_3_matched_indexes = Img_3_descriptor[0][match_ind3_2]


# fig = plt.figure("TEST 3.2 - Matching Descriptors")
# sub = fig.add_subplot(121, title='Im_1')
# plt.imshow(Img_1, cmap=plt.cm.gray)
# plt.scatter(Img_1_feature_points_x, Img_1_feature_points_y)
# sub = fig.add_subplot(122, title='Im_2')
# plt.imshow(Img_2, cmap=plt.cm.gray)
# plt.scatter(Img_2_feature_points_x, Img_2_feature_points_y)


### TEST 3.3 ###
sol4.apply_homography(np.array([[2, 3],[10, 20]]),np.array([[3, 0, 0], [0, 3, 0], [0, 0, 1]]))
ransac0 = sol4.ransac_homography(Img_1_matched_indexes, Img_2_matched_indexes, 1500, 6)
sol4.display_matches(Img_1, Img_2, Img_1_matched_indexes, Img_2_matched_indexes, ransac0[1])

ransac1 = sol4.ransac_homography(Img_2_2_matched_indexes, Img_3_matched_indexes, 1500, 6)
sol4.display_matches(Img_2, Img_3, Img_2_2_matched_indexes, Img_3_matched_indexes, ransac1[1])

### TEST 4.0 ###
H2m = sol4.accumulate_homographies([ransac0[0], ransac1[0]], 1)
H2m1 = sol4_od.accumulate_homographies([ransac0[0], ransac1[0]], 1)
print(np.allclose(H2m[:],H2m1[:]))
panorama_pic = sol4.render_panorama([Img_1, Img_2, Img_3], H2m1)


plt.show()