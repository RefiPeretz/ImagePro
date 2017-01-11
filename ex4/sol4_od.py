import numpy as np
from scipy import signal as sig
import sol4_add
import sol4_utils
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt


### Section 3.1 ###
WORK_PYR_LVL = 2
RES_PYR_LVL = 0
DESC_RAD = 3
Y_AXIS = 0
X_AXIS = 1


def harris_corner_detector(im):
    """
    Finds corner in the input image by using Haris corner detector algoritm.
    :param im: grayscale image to find key points inside.
    :return: An array with shape (N,2) of [x,y] key points locations in im.
    """
    # derivatives
    vertical_kernel = np.array([[1., 0., -1.]])
    horizontal_kernel = np.array([[1.], [0.], [-1.]])
    I_x = sig.convolve2d(im, vertical_kernel, mode='same')
    I_y = sig.convolve2d(im, horizontal_kernel, mode='same')

    # I_x2, I_y2, I_xy, I_yx
    I_x2 = sol4_utils.blur_spatial(I_x ** 2, 3)
    I_y2 = sol4_utils.blur_spatial(I_y ** 2, 3)
    I_xy = sol4_utils.blur_spatial(I_x * I_y, 3)

    # response image
    det_M = I_x2 * I_y2 - I_xy * I_xy
    trace_M = I_x2 + I_y2
    k = 0.04
    R = det_M - k * (trace_M ** 2)

    # local maximum threshold
    indexes = np.nonzero(sol4_add.non_maximum_suppression(R))
    return np.array(list(zip(indexes[1], indexes[0])))


# def sample_descriptor(im, pos, desc_rad):
#     """
#     Sample descriptors around the feature points provided. Note that sample_descriptor already expects the grayscale
#     image im to be the 3rd level pyramid image. Note also that to obtain 7 × 7 descriptors,
#     desc_rad should be set to 3.
#     :param im: grayscale image to sample within
#     :param pos: An array with shape (N,2) of [x,y] positions to sample descriptors in im.
#     :param desc_rad: − ”Radius” of descriptors to compute.
#     :return: A 3D array with shape (K,K,N) containing the ith descriptor at desc(:,:,i). The per−descriptor dimensions KxK
#              are related to the desc rad argument as follows K = 1+2∗desc rad.
#     """
#     K = 1 + 2 * desc_rad
#     K_LOW_LIM = int(K/2)
#     K_HIGH_LIM = int(K/2) + 1
#
#     # sample 7 x 7 matrix around feature points
#     descriptor_matrix = [np.meshgrid(np.arange(point[Y_AXIS] - K_LOW_LIM, point[Y_AXIS] + K_HIGH_LIM),
#                                      np.arange(point[X_AXIS] - K_LOW_LIM, point[X_AXIS] + K_HIGH_LIM),
#                                      indexing='ij') for point in pos] #TODO is there a way to optimize this with np matrix instead of python list?
#
#     # interpolate values on original image indexes
#     d = [map_coordinates(im, [descriptor[Y_AXIS].flatten(), descriptor[X_AXIS].flatten()],
#                          order=1, prefilter=False).reshape(K, K) for descriptor in descriptor_matrix] #TODO is there a way to optimize this with np matrix instead of python list?
#
#     # normalize descriptors
#     #d = [(descriptor - descriptor.mean()) / (np.linalg.norm(descriptor - descriptor.mean())) for descriptor in d] #TODO devision by zero!
#
#     for i in range(pos.shape[0]):
#         desc = d[i]
#         desc = (desc - desc.mean())
#         normalization = np.linalg.norm(desc - desc.mean())  # TODO handle the div error when zero
#         if normalization != 0:
#             desc / normalization
#         else:
#             desc *= 0
#         d[i] = desc
#
#     res = np.array(d).reshape(K, K, len(d))
#     #res[np.isnan(res)] = 0
#     return res
def sample_descriptor(im, pos, desc_rad):
    """
    Sample descriptors around the feature points provided. Note that sample_descriptor already expects the grayscale
    image im to be the 3rd level pyramid image. Note also that to obtain 7 × 7 descriptors,
    desc_rad should be set to 3.
    :param im: grayscale image to sample within
    :param pos: An array with shape (N,2) of [x,y] positions to sample descriptors in im.
    :param desc_rad: − ”Radius” of descriptors to compute.
    :return: A 3D array with shape (K,K,N) containing the ith descriptor at desc(:,:,i). The per−descriptor dimensions KxK
             are related to the desc rad argument as follows K = 1+2∗desc rad.
    """
    K = 1 + 2 * desc_rad

    desc = np.empty((K, K, pos.shape[0]))

    # calculate descriptor for each point
    for i in range(pos.shape[0]):
        y = np.arange(pos[i, 0] - 3, pos[i, 0] + 4)  # y axis coordinates
        x = np.arange(pos[i, 1] - 3, pos[i, 1] + 4)  # x axis coordinates

        yv, xv = np.meshgrid(y, x, indexing='ij')

        # interpolate the values of the descriptor
        d = map_coordinates(im, [xv.T, yv.T], order=1, prefilter=False)

        # normalize
        d_work = (d - np.mean(d))
        normalization = np.linalg.norm(d - np.mean(d))
        if normalization != 0:
            d_work /= normalization
        else:
            d_work *= 0

        # add to descriptors data structure
        desc[:, :, i] = d_work

    return desc


def find_features(pyr):
    """
    Detects feature points in the pyramid and samples their descriptors.
    :param pyr: Gaussian pyramid of a grayscale image having 3(=WORK_PYR_LVL) levels.
    :return: pos − An array with shape (N,2) of [x,y] feature location per row found in the (third pyramid level of the) image. These
                   coordinates are provided at the pyramid level pyr[0](=RES_PYR_LVL).
             desc − A feature descriptor array with shape (K,K,N).
    """
    feature_points_coordinates_res = sol4_add.spread_out_corners(pyr[RES_PYR_LVL], 7, 7, 12)
    feature_points_coordinates_work = (2 ** (RES_PYR_LVL - WORK_PYR_LVL)) * feature_points_coordinates_res
    feature_points_descriptors = sample_descriptor(pyr[WORK_PYR_LVL],
                                                   feature_points_coordinates_work,
                                                   DESC_RAD)

    return feature_points_coordinates_res, feature_points_descriptors


### Section 3.2 ###
def match_features(desc1, desc2, min_score):
    """
    Matches descriptors between desc1 and desc2 according to the algorithm given in the exercise.
    :param desc1: A feature descriptor array with shape (K,K,N1).
    :param desc2: A feature descriptor array with shape (K,K,N2).
    :param min_score: Minimal match score between two descriptors required to be regarded as corresponding points.
    :return: match ind1 − Array with shape (M,) and dtype int of matching indices in desc1.
             match ind2 − Array with shape (M,) and dtype int of matching indices in desc2.
    """
    # calculate scores matrix
    scores_matrix = desc1.reshape((desc1.shape[0] ** 2, desc1.shape[2])).T\
        .dot(desc2.reshape((desc2.shape[0] ** 2, desc2.shape[2])))

    # find two maximums in each row
    rows_two_max = scores_matrix.T >= np.partition(scores_matrix.T, scores_matrix.shape[1] - 2, axis=0)[:][-2]

    # find two maximums in each col
    cols_two_max = scores_matrix >= np.partition(scores_matrix, scores_matrix.shape[0] - 2, axis=0)[:][-2]

    # return 2 arrays of indexes of rows and cols, meaning indexes in desc1 and desc2 accordingly.
    return np.nonzero((scores_matrix > min_score) & rows_two_max.T & cols_two_max)


### Section 3.3 ###
def apply_homography(pos1, H12):
    """
    Applies a homography transformation on a set of points.
    :param pos1: An array with shape (N,2) of [x,y] point coordinates.
    :param H12: A 3x3 homography matrix.
    :return: pos2 − An array with the same shape as pos1 with [x,y] point coordinates in image i+1 obtained from
                    transforming pos1 using H12.
    """
    pos1_homographic = np.insert(pos1, 2, 1, axis=1).T
    pos1_work = H12.dot(pos1_homographic)
    pos2 = np.array([pos1_work[0] / pos1_work[2], pos1_work[1] / pos1_work[2]], dtype=pos1.dtype).T
    return pos2


def ransac_homography(pos1, pos2, num_iter, inlier_tol):
    """
    Performs RANSAC (Random SAmple Consensus) homography fitting based on the euclidiean distance
    of the homographed points.
    :param pos1: Array with shape (N,2) containing n rows of [x,y] coordinates of matched points.
    :param pos2: Array with shape (N,2) containing n rows of [x,y] coordinates of matched points.
    :param num_iter: Number of RANSAC iterations to perform.
    :param inlier_tol: inlier tolerance threshold.
    :return: H12 − A 3x3 normalized homography matrix.
             inliers − An Array with shape (S,) where S is the number of inliers, containing the indices
                       in pos1/pos2 of the maximal set of inlier matches found.
    """
    # find best homography for biggest set of inliers
    max_inliers_idx_set = []
    max_inliers_seen = 0
    for i in range(num_iter):
        four_rand_idx = np.random.permutation(np.arange(pos1.shape[0]))[:4]
        pos1_points = np.take(pos1, four_rand_idx, axis=0)
        pos2_points = np.take(pos2, four_rand_idx, axis=0)
        H12 = sol4_add.least_squares_homography(pos1_points, pos2_points)        #TODO should i normalize H12?
        if H12 is not None:
            pos2_homography = apply_homography(pos1, H12)
            #euclidiean_distance = np.sqrt(np.sum((pos2_homography-pos2)**2, axis=-1))
            euclidiean_distance = np.linalg.norm(pos2_homography-pos2, axis=1)
            inliers_idx = np.where(euclidiean_distance < inlier_tol)[0]
            if len(inliers_idx) >= max_inliers_seen:
                max_inliers_seen = len(inliers_idx)
                max_inliers_idx_set = inliers_idx

    # calculate final homography
    pos1_points = np.take(pos1, max_inliers_idx_set, axis=0)
    pos2_points = np.take(pos2, max_inliers_idx_set, axis=0)
    H12 = sol4_add.least_squares_homography(pos1_points, pos2_points)
    return H12, max_inliers_idx_set


def display_matches(im1, im2, pos1, pos2, inliers):
    """
    Visualizes the full set of point matches and the inlier matches detected by RANSAC.
    :param im1: grayscale image
    :param im2: grayscale image
    :param pos1: Array with shape (N,2), containing N rows of [x,y] coordinates of matched points in im1
                (i.e. the match of the ith coordinate is pos1[i,:] in im1).
    :param pos2: Array with shape (N,2), containing N rows of [x,y] coordinates of matched points in im2
                (i.e. the match of the ith coordinate is pos1[i,:] in im2).
    :param inliers: An array with shape (S,) of inlier matches
    """
    # connect images and mark matching points as red dots
    concatenated_image = np.hstack((im1, im2))
    fig = plt.figure()
    plt.axis('off')
    plt.imshow(concatenated_image, cmap=plt.cm.gray)

    # plot dots matched on concatenated_image
    plt.scatter(pos1[:, 0], pos1[:, 1], color='r')
    plt.scatter(pos2[:, 0] + im1.shape[1], pos2[:, 1], color='r')

    # plot blue lines between outliers
    outliers = np.arange(0, pos1.shape[0])
    outliers[inliers] = -1
    outliers = np.where(outliers != -1)

    plt.plot((pos1[outliers][:, 0], pos2[outliers][:, 0] + im1.shape[1]),
             (pos1[outliers][:, 1], pos2[outliers][:, 1]), mfc='r', c='b', lw=.4, ms=5, marker='o')

    # plot yellow lines between inliers
    plt.plot((pos1[inliers][:, 0], pos2[inliers][:, 0] + im1.shape[1]),
             (pos1[inliers][:, 1], pos2[inliers][:, 1]), mfc='r', c='y', lw=.4, ms=0.5, marker='o')

def accumulate_homographies(H_successive, m):
    """
    Calculates cumulative homographies matrices.
    :param H_successive: A list of M−1 3x3 homography matrices where H_successive[i] is a homography
                         that transforms points from coordinate system i to coordinate system i+1.
    :param m: Index of the coordinate system we would like to accumulate the given homographies towards.
    :return: H2m − A list of M 3x3 homography matrices, where H2m[i] transforms points from coordinate system i to
             coordinate system m. homography matrices should always maintain the property that H[2,2]==1, so each
             matrix is normalized.
    """
    H2m = []
    for i in range(len(H_successive) + 1):
        work_mat = [[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]]
        if i < m:
            for j in range(m - 1, i - 1, -1):
                work_mat *= H_successive[j]
        elif i > m:
            for j in range(m, i, 1):
                work_mat *= np.linalg.inv(H_successive[j])
        else:
            work_mat = np.eye(3)
        work_mat /= work_mat[2, 2]
        H2m.append(work_mat)
    return H2m


def render_panorama(ims, Hs):
    """
    Renders a grayscale panorama.
    :param ims: A list of grayscale images. (Python list)
    :param Hs: A list of 3x3 homography matrices. Hs[i] is a homography that transforms points from the coordinate
               system of ims [i] to the coordinate system of the panorama. (Python list)
    :return: panorama − A grayscale panorama image composed of vertical strips, backwarped using homographies from Hs,
             one from every image in ims.
    """
    # calculate 4 corners of each image
    corners_idx = np.array([[0, 0], [ims[0].shape[1], 0], [0, ims[0].shape[0]], [ims[0].shape[1], ims[0].shape[0]]])
    ims_corners_homographed = np.array([apply_homography(corners_idx, H).T for H in Hs]) # TODO can this be vectorised?

    # find 4 corners of panorama according to min\max homographed values
    x_max = np.max(ims_corners_homographed[:, 0])
    x_min = np.min(ims_corners_homographed[:, 0])
    y_max = np.max(ims_corners_homographed[:, 1])
    y_min = np.min(ims_corners_homographed[:, 1])

    # create canvas
    panorama = np.zeros((y_max - y_min + 1, x_max - x_min + 1))

    # calculate image stripes borders
    image_center_idx = np.array([[np.round(ims[0].shape[1] / 2), np.round(ims[0].shape[0] / 2)]]) #TODO should i round or not? there is no .5 pixel
    images_centers = [apply_homography(image_center_idx, H).T for H in Hs] # TODO can this be vectorised?
    stripes_borders = []
    for i in range(len(images_centers) - 1):
        stripes_borders.append(((images_centers[i][0] + images_centers[i+1][0]) / 2)) #TODO should i round or not? there is no .5 pixel

    # find images x-coordinates and y_coordinates on the panorama
    for i in range(len(ims)):
        if i == 0:
            x = np.arange(x_min, stripes_borders[i])  # x axis coordinates
        elif i == (len(ims) - 1):
            x = np.arange(stripes_borders[i], x_max)  # x axis coordinates
        else:
            x = np.arange(stripes_borders[i], stripes_borders[i+1])  # x axis coordinates

        y = np.arange(y_min, y_max)  # y axis coordinates

        yv, xv = np.meshgrid(y, x, indexing='ij')

        # interpolate the values of the descriptor
        #d = map_coordinates(im, [xv.T, yv.T], order=1, prefilter=False)
