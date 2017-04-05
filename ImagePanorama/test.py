import numpy as np
import sol4_add
import sol4_utils
import matplotlib.pyplot as plt
import random
from scipy.ndimage.filters import convolve as convolve
from scipy.ndimage.interpolation import map_coordinates as map_coordinates


def harris_corner_detector(im):
    div = (np.array([-1, 0, 1])).reshape(1, 3)
    Ix = convolve(im, div)
    Iy = convolve(im, np.transpose(div))
    IxIx = sol4_utils.blur_spatial(Ix * Ix, 3)
    IxIy = sol4_utils.blur_spatial(Ix * Iy, 3)
    IyIy = sol4_utils.blur_spatial(Iy * Iy, 3)
    k = 0.04
    M = np.dstack((IxIx, IxIy, IxIy, IyIy))
    M = M.reshape(M.shape[0], M.shape[1], 2, 2)
    R = np.linalg.det(M[:, :]) - k * (np.trace(M, axis1=2, axis2=3) ** 2)
    ret = np.dstack(np.where(sol4_add.non_maximum_suppression(R.transpose())))
    return ret.reshape(ret.shape[1], 2)


def sample_descriptor(im, pos, desc_rad):
    k = 1 + 2 * desc_rad
    i = 0
    desc = np.zeros((k, k, pos.shape[0]))
    pos = 0.25 * pos
    for p in pos:
        X, Y = np.meshgrid(np.arange(p[0] - desc_rad, p[0] + desc_rad + 1),
                           np.arange(p[1] - desc_rad, p[1] + desc_rad + 1))
        pos_new = (map_coordinates(im, np.array([Y.flatten(), X.flatten()]), order=1, prefilter=False)).reshape((k, k))
        mean = np.mean(pos_new)
        if np.linalg.norm(pos_new - mean) == 0:
            desc[:, :, i] = (pos_new - mean)
        else:
            desc[:, :, i] = (pos_new - mean) / (np.linalg.norm(pos_new - mean))
        i += 1
    return desc


def find_features(pyr):
    desc_rad = 3
    pos = sol4_add.spread_out_corners(pyr[0], 7, 7, desc_rad * 4)
    desc = sample_descriptor(pyr[2], pos, desc_rad)
    return pos, desc


def calc_s_matrix(desc1, desc2):
    d1 = np.transpose(desc1.reshape((desc1.shape[0] * desc1.shape[1], desc1.shape[2])))
    d2 = desc2.reshape((desc2.shape[0] * desc2.shape[1], desc2.shape[2]))
    return d1.dot(d2)


def calc_second_best(S):
    arg_sort = np.argsort(S, axis=1)
    second = arg_sort[:, -2]
    return S[np.arange(S.shape[0]), second]


def match_features(desc1, desc2, min_score):
    match_ind1 = []
    match_ind2 = []
    S = calc_s_matrix(desc1, desc2)
    SBi = calc_second_best(S)
    SBj = calc_second_best(np.transpose(S))
    for i in range(desc1.shape[2]):
        for j in range(desc2.shape[2]):
            if S[i, j] > min_score:
                if ((S[i, j] >= SBi[i]) and (S[i, j] >= SBj[j])):
                    match_ind1.append(i)
                    match_ind2.append(j)
    return np.array(match_ind1), np.array(match_ind2)


def apply_homography(pos1, H12):
    pos = np.ones((len(pos1), 3))
    pos[:, 0:2] = pos1
    p = H12.dot(np.transpose(pos))
    pos2 = np.divide(p[:2, :], p[2, :])
    return np.transpose(pos2)


def random_points(s):
    points = np.arange(s)
    np.random.shuffle(points)
    return points[:4]


def apply_and_get_nunber_of_inliers(H, pos1, pos2, inlier_tol):
    P2 = apply_homography(pos1, H)
    E = np.linalg.norm(P2 - pos2, axis=1) ** 2
    return (np.where(E < inlier_tol))[0]


def ransac_homography(pos1, pos2, num_iter, inlier_tol):
    E = np.array([])
    pointsArr = []
    max_inliers = 0
    for i in range(num_iter):
        points = random_points(pos1.shape[0])
        pointsArr.append(points)
        H = sol4_add.least_squares_homography(pos1[points], pos2[points])
        if H is None:
            continue
        inliers = apply_and_get_nunber_of_inliers(H, pos1, pos2, inlier_tol)
        if inliers.size > max_inliers:
            max_inliers = inliers.size
            E = inliers
    H = sol4_add.least_squares_homography(pos1[E], pos2[E])
    inliers = apply_and_get_nunber_of_inliers(H, pos1, pos2, inlier_tol)
    return H, inliers


def display_matches(im1, im2, pos1, pos2, inliers):
    plt.figure()
    plt.imshow(np.hstack((im1, im2)), cmap=plt.cm.gray)
    for i in range(len(pos1)):
        color = 'y' if i in inliers else 'b'
        plt.plot([pos1[i, 0], pos2[i, 0] + im1.shape[1]], [pos1[i, 1], pos2[i, 1]], mfc='r', c=color, lw=1, ms=5,
                 marker='o')
    plt.show()


def accumulate_homographies(H_successive, m):
    H2m = []
    if m > 0:
        H = H_successive[m - 1]
        H2m.insert(0, H)
        for i in range(m - 2, -1, -1):
            H = np.dot(H, H_successive[i])
            H = H / H[2, 2]
            H2m.insert(0, H)
    H2m.append(np.eye(3))
    H = np.linalg.inv(H_successive[m])
    H = H / H[2, 2]
    H2m.append(H)
    for i in range(m + 1, len(H_successive), 1):
        H = np.dot(H, np.linalg.inv(H_successive[i]))
        H = H / H[2, 2]
        H2m.append(H)
    return H2m


def calc_max_min_corners(ims, Hs):
    p = np.empty((0, 2))
    for i in range(len(Hs)):
        s = ims[i].shape
        corners_pos = np.array([[0, 0], [s[1] - 1, 0], [0, s[0] - 1], [s[1] - 1, s[0] - 1]])
        p = np.append(p, apply_homography(corners_pos, Hs[i]), axis=0)
    min_cor = np.array([np.min(p[:, 0]), np.min(p[:, 1])])
    max_cor = np.array([np.max(p[:, 0]), np.max(p[:, 1])])
    return min_cor, max_cor


def calc_centers(ims, Hs):
    centers = np.zeros((len(ims), 2))
    for i in range(len(Hs)):
        s = ims[i].shape
        centers[i, :] = (apply_homography(np.array([[(s[1] - 1) / 2, (s[0] - 1) / 2]]), Hs[i]))
    return centers


def render_panorama(ims, Hs):
    min_cor, max_cor = calc_max_min_corners(ims, Hs)
    centers = calc_centers(ims, Hs)
    X, Y = np.meshgrid(np.arange(min_cor[0], max_cor[0] + 1), np.arange(min_cor[1], max_cor[1] + 1))
    panorama = np.zeros(X.shape)
    border = [0]
    for i in range(len(ims) - 1):
        border.append(np.round(((centers[i, 0] + centers[i + 1, 0]) / 2) - min_cor[0]))
    border.append(panorama.shape[1])
    for i in range(len(ims)):
        left = int(border[i])
        right = int(border[i + 1])
        x_new = X[:, left:right]
        y_new = Y[:, left:right]
        ind = np.array(apply_homography(np.transpose([x_new.flatten(), y_new.flatten()]), np.linalg.inv(Hs[i])))
        mapped = map_coordinates(ims[i], [ind[:, 1], ind[:, 0]], order=1, prefilter=False)
        panorama[:, left:right] = mapped.reshape(panorama[:, left:right].shape)
    # panorama = blend(panorama, mapped.reshape(panorama[:,left:right].shape) , left, right)
    return panorama


def blend(panorama, part, left, right):
    size = (np.around(np.log2(panorama.shape))).astype(np.int32)
    s = 2 ** np.max(size)
    mask = np.ones((s, s))
    mask[:, left:right] = 0
    panorama_new = np.zeros(mask.shape)
    panorama_new[:panorama.shape[0], :panorama.shape[1]] = panorama
    panorama_part = np.zeros(mask.shape)
    panorama_part[:part.shape[0], left:right] = part
    p = sol4_utils.pyramid_blending(panorama_new, panorama_part, mask, 7, 3, 3)
    return p[:panorama.shape[0], :panorama.shape[1]]


im1 = sol4_utils.read_image('external/backyard1.jpg', 1)
im2 = sol4_utils.read_image('external/backyard2.jpg', 1)
im3 = sol4_utils.read_image('external/backyard3.jpg', 1)

pyr1, filter_vec1 = sol4_utils.build_gaussian_pyramid(im1, 3, 3)
pos1, desc1 = find_features(pyr1)

pyr2, filter_vec2 = sol4_utils.build_gaussian_pyramid(im2, 3, 3)
pos2, desc2 = find_features(pyr2)

pyr3, filter_vec3 = sol4_utils.build_gaussian_pyramid(im3, 3, 3)
pos3, desc3 = find_features(pyr3)

match_ind1, match_ind2 = match_features(desc1, desc2, 0)
p1 = pos1[match_ind1]
p2 = pos2[match_ind2]
H12, inliers12 = ransac_homography(p1, p2, 500, 6)
# display_matches(im1,im2,p1,p2,inliers12)

match_ind3, match_ind4 = match_features(desc2, desc3, 0)
p3 = pos2[match_ind3]
p4 = pos3[match_ind4]
H23, inliers23 = ransac_homography(p3, p4, 500, 6)

H = np.array([H12, H23])

Ha = accumulate_homographies(H, 1)
ims = [im1, im2, im3]
a = render_panorama(ims, Ha)


# def render_panorama1(ims, Hs):
#     x_val,y_val = [],[]
#     centers_ims = []
#
#     center_dummy = np.array([[int((ims[0].shape[1] - 1 )/ 2), int((ims[0].shape[0]  - 1)/ 2)]])
#     corners = np.array([[0,0],[ims[0].shape[1] - 1,0],[0, ims[0].shape[0] - 1],[ims[0].shape[1] - 1,ims[0].shape[0] - 1]])
#     for i in range(len(ims)):
#         cur_pos = apply_homography(np.copy(corners), Hs[i])
#         x_val += list(cur_pos[:,0])
#         y_val += list(cur_pos[:,1])
#         centers_ims.append(apply_homography(np.copy(center_dummy), Hs[i])[:])
#     x_max,x_min = max(x_val), min(x_val)
#     y_max,y_min = max(y_val), min(y_val)
#
#     xs, ys = np.meshgrid(np.arange(x_min, x_max + 1), np.arange(y_min, y_max + 1))
#     panorama = np.zeros((xs.shape[0],xs.shape[1]))
#
#     M_set = []
#     M_set.append(0)
#     for i in range(len(ims) - 1):
#         M_set.append(np.round(((centers_ims[i][0][0] + centers_ims[i + 1][0][0]) / 2) - x_min))
#     M_set.append(panorama.shape[1])
#
#     border = M_set
#
#     for i in range(len(ims)):
#         left = int(border[i])
#         right = int(border[i + 1])
#         x_new = xs[:, left:right]
#         y_new = ys[:, left:right]
#         ind = np.array(apply_homography(np.transpose([x_new.flatten(), y_new.flatten()]), np.linalg.inv(Hs[i])))
#         mapped = map_coordinates(ims[i], [ind[:, 1], ind[:, 0]], order=1, prefilter=False)
#         panorama[:, left:right] = mapped.reshape(panorama[:, left:right].shape)
#     # panorama = blend(panorama, mapped.reshape(panorama[:,left:right].shape) , left, right)
#     return panorama