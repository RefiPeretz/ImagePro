import sol4_utils as util
import sol4_add as add
import numpy as np
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt
#TODO delete
import sol4_od
from functools import reduce
from scipy.ndimage.filters import convolve as convolve



def calculate_matrix(im):
    der_x_im, der_y_im = util.conv_der(im)
    matrix_1 = util.blur_spatial(der_x_im * der_x_im, 3)
    matrix_2 = util.blur_spatial(der_x_im * der_y_im, 3)
    matrix_3 = util.blur_spatial(der_y_im * der_x_im, 3)
    matrix_4 = util.blur_spatial(der_y_im * der_y_im, 3)
    return matrix_1,matrix_2,matrix_3,matrix_4

def harris_corner_detector(im):
    matrix_1,matrix_2,matrix_3,matrix_4 = calculate_matrix(im)
    k = 0.04
    tmp1 = (matrix_1*matrix_4 - matrix_2*matrix_3)
    tmp2 = k*((matrix_1 + matrix_4)**2)
    R = tmp1 - tmp2
    R = R.T
    #TODO maybe change
    res_r = np.dstack(np.where(add.non_maximum_suppression(R)))
    return res_r.reshape(res_r.shape[1], 2)

#TODO delete
# def harris_corner_detector1(im):
#     matrix_1,matrix_2,matrix_3,matrix_4 = calculate_matrix(im)
#     k = 0.04
#     M = np.dstack((matrix_1, matrix_2, matrix_2, matrix_4))
#     M = M.reshape(M.shape[0], M.shape[1], 2, 2)
#     R = np.linalg.det(M[:, :]) - k * (np.trace(M, axis1=2, axis2=3) ** 2)
#     ret = np.dstack(np.where(add.non_maximum_suppression(R.transpose())))
#     return ret.reshape(ret.shape[1], 2)
#
#TODO delete
# def harris_corner_detector1(im):
#     div = (np.array([-1, 0, 1])).reshape(1, 3)
#     Ix = convolve(im, div)
#     Iy = convolve(im, np.transpose(div))
#     IxIx = util.blur_spatial(Ix * Ix, 3)
#     IxIy = util.blur_spatial(Ix * Iy, 3)
#     IyIy = util.blur_spatial(Iy * Iy, 3)
#     k = 0.04
#     M = np.dstack((IxIx, IxIy, IxIy, IyIy))
#     M = M.reshape(M.shape[0], M.shape[1], 2, 2)
#     R = np.linalg.det(M[:, :]) - k * (np.trace(M, axis1=2, axis2=3) ** 2)
#     ret = np.dstack(np.where(add.non_maximum_suppression(R.transpose())))
#     return ret.reshape(ret.shape[1], 2)

def calculate_patch(cor,K,border):
    x,y = cor[0],cor[1]
    xs = np.meshgrid(np.arange(x - border, x + border+1), np.arange(x - border, x + border+1))[1].reshape(1,K*K)
    ys = np.meshgrid(np.arange(y - border, y + border+1), np.arange(y - border, y + border+1))[0].reshape(1,K*K)
    return  np.array([ys[0],xs[0]])

def sample_descriptor(im, pos, desc_rad):
    #TODO should we calculate it?
    reduce_factor = 2 ** -2
    K = 2*desc_rad + 1
    border = desc_rad
    desc_list = []
    for cor in pos:
        papo = calculate_patch(reduce_factor*cor,K,border)
        res_patch = map_coordinates(im, papo, order=1, prefilter=False).reshape(K,K)
        #TODO normlize check if done correct?
        divide_with = (np.linalg.norm(res_patch -  np.mean(res_patch)))
        if(divide_with != 0):
            res_patch = (res_patch - np.mean(res_patch)) / divide_with
            desc_list.append(res_patch)
        else:
            desc_list.append(res_patch - res_patch)

    return np.dstack(desc_list)



def find_features(pyr):

    pos = add.spread_out_corners(pyr[0],7,7,12)
    #TODO should we use 3?
    return pos,sample_descriptor(pyr[2],pos,3)


def calculate_desc_matrix(desc1,desc2):
    return np.dot(np.transpose(desc1.reshape((desc1.shape[0] * desc1.shape[1], desc1.shape[2]))),\
                  desc2.reshape((desc2.shape[0] * desc2.shape[1], desc2.shape[2])))



def match_features(desc1,desc2,min_score):
    scores_matrix = desc1.reshape((desc1.shape[0] ** 2, desc1.shape[2])).T\
        .dot(desc2.reshape((desc2.shape[0] ** 2, desc2.shape[2])))

    # find two maximums in each row
    rows_two_max = scores_matrix.T >= np.partition(scores_matrix.T, scores_matrix.shape[1] - 2, axis=0)[:][-2]

    # find two maximums in each col
    cols_two_max = scores_matrix >= np.partition(scores_matrix, scores_matrix.shape[0] - 2, axis=0)[:][-2]

    # return 2 arrays of indexes of rows and cols, meaning indexes in desc1 and desc2 accordingly.
    return np.nonzero((scores_matrix >= min_score) & rows_two_max.T & cols_two_max)

    score_matrix = calculate_desc_matrix(desc1,desc2)
    score_matrix_t = score_matrix.T
    max_values_desc1 = np.partition(score_matrix_t,score_matrix_t.shape[0] - 2, axis=0)[:][-2]
    max_values_desc2 = np.partition(score_matrix, score_matrix[0] - 2, axis=0)[:][-2]

    bool_greater_minscore_desc1 =  score_matrix_t >= min_score
    bool_greater_secbest_desc1 = score_matrix_t >= max_values_desc1
    desc1_choose_bool = bool_greater_minscore_desc1 & bool_greater_secbest_desc1

    bool_greater_minscore_desc2 =  score_matrix >= min_score
    bool_greater_secbest_desc2 = score_matrix >= max_values_desc2
    desc2_choose_bool = bool_greater_minscore_desc2 & bool_greater_secbest_desc2

    final_matrix_pairs_index =  desc1_choose_bool.T & desc2_choose_bool
    return np.nonzero(final_matrix_pairs_index)

    # papo2 = desc1[:,:,5].flatten()
    # papo3 = desc2[:, :,159].flatten()
    # papo4 = np.dot(papo2,papo3)

# def apply_homography(pos1, H12):
#     work_set =np.c_[pos1, np.ones(pos1.shape[0],dtype=int)]
#     res_matrix = np.dot(H12,work_set.T).reshape(pos1.shape[0],pos1.shape[1] + 1)
#     divide = res_matrix[:,-1:]
#     res_matrix /= divide
#     pp = res_matrix[:, :-1]
#     return (res_matrix[:,:-1])




def apply_homography(pos1, H12):
    work_set = np.c_[pos1, np.ones(pos1.shape[0], dtype=int)]
    res_mat = np.dot(H12, work_set.T)
    res_mat = (res_mat / res_mat[-1:,]).T
    return (res_mat[:,:-1]).round()

def ransac_homography(pos1,pos2,num_iters,inlier_tol):
    max_inliers = 0
    final_inliers_set = None
    for i in range(num_iters):
        choesn_points = np.random.randint(pos1.shape[0], size=4)
        match_index_im1 , match_index_im2  = pos1[choesn_points] , pos2[choesn_points]
        cur_H12 = add.least_squares_homography(match_index_im1, match_index_im2)
        if(cur_H12 != None):
            transformed_pos1 = apply_homography(np.copy(pos1), cur_H12)
            e_set = np.linalg.norm((pos2 - transformed_pos1), 2, axis=1)
            cur_inliers = np.argwhere(e_set < inlier_tol)
            if cur_inliers.size > max_inliers:
                max_inliers = cur_inliers.size
                final_inliers_set = np.copy(cur_inliers)

    return add.least_squares_homography(pos1[final_inliers_set.flatten()], pos2[final_inliers_set.flatten()]), final_inliers_set.flatten()


def display_matches(im1,im2,pos1,pos2,inliers):

    linked_im = np.hstack((im1, im2))


    plt.scatter(pos1[:, 0], pos1[:, 1], color='r')
    plt.scatter(im1.shape[1] + pos2[:, 0], pos2[:, 1], color='r')

    outliers = np.delete(np.arange(pos1.shape[0]),inliers,axis=0)

    plt.plot((pos1[outliers][:, 0], im1.shape[1] + pos2[outliers][:,0]), \
        (pos1[outliers][:, 1], pos2[outliers][:, 1]), color='b', ms=0.5 )

    plt.plot((pos1[inliers][:, 0], im1.shape[1] + pos2[inliers][:,0]), \
        (pos1[inliers][:, 1], pos2[inliers][:, 1]), color='y', ms=0.5 )

    plt.imshow(linked_im, cmap=plt.cm.gray)

    plt.show()


#
# def accumulate_homographies(H_successive,m):
#     #TODO change
#     H2m = []
#     for i in range(len(H_successive) + 1):
#         res_matrix = np.ones((3,3))
#         if i > m:
#             for j in range(m, i):
#                 res_matrix *= np.linalg.inv(H_successive[j])
#         elif i < m:
#
#             for j in range(m - 1, i - 1, -1):
#                 res_matrix *= H_successive[j]
#         else:
#             res_matrix = np.eye(3)
#         res_matrix /= res_matrix[2, 2]
#         H2m.append(res_matrix)
#
#     return H2m
#
#
def render_panorama1(ims,H):
    #max_x,max_y,min_x,min_y = [],[],[],[]
    x_val,y_val = [],[]
    centers_ims = []

    center_dummy = np.array([[int(ims[0].shape[0] / 2), int(ims[0].shape[1] / 2)]])
    corners = np.array([[0,0],[ims[0].shape[1] - 1,0],[0, ims[0].shape[0] - 1],[ims[0].shape[1] - 1,ims[0].shape[0] - 1]])
    for i in range(len(ims)):
        cur_pos = apply_homography(np.copy(corners), H[i])
        x_val += list(cur_pos[:,0])
        y_val += list(cur_pos[:,1])
        centers_ims.append(apply_homography(np.copy(center_dummy), H[i])[:,0])
    x_max,x_min = max(x_val), min(x_val)
    y_max,y_min = max(y_val), min(y_val)


    M_set = []
    for i in range(len(centers_ims) - 1):
        M_set.append((centers_ims[i] + centers_ims[i+1])/2)


    pan = np.zeros((x_max - x_min + 1 , y_max - y_min + 1))
#
#


def accumulate_homographies(H_successive,m):
    #TODO change
    H2m = []
    for i in range(len(H_successive) + 1):
        res_matrix = np.eye(3)
        if i > m:
            for j in range(m, i):
                res_matrix = res_matrix.dot(np.linalg.inv(H_successive[j]))
        elif i < m:

            for j in range(m - 1, i - 1, -1):
                res_matrix = res_matrix.dot(H_successive[j])
        else:
            res_matrix = np.eye(3)
        res_matrix /= res_matrix[2, 2]
        H2m.append(res_matrix)

    return H2m

def accumulate_homographies1(H_successive, m):
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
        work_mat = np.eye(3)
        if i < m:
            for j in range(m - 1, i - 1, -1):
                work_mat = np.dot(work_mat, H_successive[j])
        elif i > m:
            for j in range(m, i, 1):
                work_mat = np.dot(work_mat, np.linalg.inv(H_successive[j]))
        else:
            work_mat = np.eye(3)
        work_mat /= work_mat[2][2]
        H2m.append(work_mat)
    return H2m




def render_panorama(ims, Hs):
    x_val,y_val = [],[]
    centers_ims = []

    center_dummy = np.array([[int((ims[0].shape[1] - 1 )/ 2), int((ims[0].shape[0]  - 1)/ 2)]])
    corners = np.array([[0,0],[ims[0].shape[1] - 1,0],[0, ims[0].shape[0] - 1],[ims[0].shape[1] - 1,ims[0].shape[0] - 1]])
    for i in range(len(ims)):
        cur_pos = apply_homography(np.copy(corners), Hs[i])
        x_val += list(cur_pos[:,0])
        y_val += list(cur_pos[:,1])
        centers_ims.append(apply_homography(np.copy(center_dummy), Hs[i])[:])
    x_max,x_min = max(x_val), min(x_val)
    y_max,y_min = max(y_val), min(y_val)

    xs, ys = np.meshgrid(np.arange(x_min, x_max + 1), np.arange(y_min, y_max + 1))
    panorama = np.zeros((xs.shape[0],xs.shape[1]))

    M_set = []
    M_set.append(0)
    for i in range(len(ims) - 1):
        M_set.append(int(np.round(((centers_ims[i][0][0] + centers_ims[i + 1][0][0]) / 2) - x_min)))
    M_set.append(panorama.shape[1])

    border = M_set

    for i in range(len(ims)):
        cur_grid_x, cur_grid_y = xs[:, border[i]:border[i+1]], ys[:, border[i]:border[i+1]]
        homographed_points = \
            np.array(apply_homography(np.dstack([cur_grid_x.flatten(), cur_grid_y.flatten()])[0], np.linalg.inv(Hs[i])))
        panorama[:, border[i]:border[i+1]] = \
            map_coordinates(ims[i], [homographed_points[:, 1], homographed_points[:, 0]], order=1, prefilter=False)\
            .reshape(panorama[:, border[i]:border[i+1]].shape)
    # panorama = blend(panorama, mapped.reshape(panorama[:,left:right].shape) , left, right)
    return panorama










        # Img_1 = util.read_image("external/oxford1.jpg", 1)
# Img_2 = util.read_image("external/oxford2.jpg", 1)
#
# Img_1_descriptor = find_features(util.build_gaussian_pyramid(Img_1, 3, 3)[0])  # TODO filter_size?
# Img_2_descriptor = find_features(util.build_gaussian_pyramid(Img_2, 3, 3)[0])  # TODO filter_size?
#
# min_score = 0.5
# match_ind1, match_ind2 = match_features(Img_1_descriptor[1], Img_2_descriptor[1], min_score)
# Img_1_matched_indexes = Img_1_descriptor[0][match_ind1]
# Img_1_feature_points_x = [x[0] for x in Img_1_matched_indexes]
# Img_1_feature_points_y = [y[1] for y in Img_1_matched_indexes]
# Img_2_matched_indexes = Img_2_descriptor[0][match_ind2]
# Img_2_feature_points_x = [x[0] for x in Img_2_matched_indexes]
# Img_2_feature_points_y = [y[1] for y in Img_2_matched_indexes]
# papo = ransac_homography(Img_1_matched_indexes,Img_2_matched_indexes,1000,10)
# display_matches(Img_1,Img_2,Img_1_matched_indexes,Img_2_matched_indexes,papo[1])
# # fig = plt.figure("TEST 3.2 - Matching Descriptors")
# # sub = fig.add_subplot(121, title='Im_1')
# # plt.imshow(Img_1, cmap=plt.cm.gray)
# # plt.scatter(Img_1_feature_points_x, Img_1_feature_points_y)
# # sub = fig.add_subplot(122, title='Im_2')
# # plt.imshow(Img_2, cmap=plt.cm.gray)
# # plt.scatter(Img_2_feature_points_x, Img_2_feature_points_y)
# # plt.show()