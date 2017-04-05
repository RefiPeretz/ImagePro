#############################################################
# FILE: sol4.py
# WRITER: Liav Steinberg
# EXERCISE : Image Processing ex4
#############################################################

import numpy as np
import sol4_utils as sut
import sol4_add as sad
import matplotlib.pyplot as plt
import scipy.signal as sg
from itertools import product
from scipy.ndimage import map_coordinates


# --------------------------3.1-----------------------------#

def harris_corner_detector(im):
    #############################################################
    # implements harris method for corner detection
    #############################################################
    dx = np.array([[1, 0, -1]])
    dy = dx.transpose()
    Ix = sg.convolve2d(im, dx, mode="same")
    Iy = sg.convolve2d(im, dy, mode="same")

    Ix_blur = sut.blur_spatial(Ix ** 2, 3)
    Iy_blur = sut.blur_spatial(Iy ** 2, 3)
    IxIy_blur = sut.blur_spatial(Ix * Iy, 3)
    # compute determinant and trace of M
    det = Ix_blur * Iy_blur - IxIy_blur ** 2
    tr = Ix_blur + Iy_blur
    R = det - 0.04 * (tr ** 2)
    return np.transpose(np.nonzero(sad.non_maximum_suppression(R)))


def sample_descriptor(im, pos, desc_rad):
    #############################################################
    # descriptor sampling
    #############################################################
    K = 1 + (2 * desc_rad)
    window_range = K >> 1
    desc = np.zeros((K, K, pos.shape[0]), dtype=np.float64)
    for idx in range(len(pos)):
        x, y = pos[idx][0].astype(np.float32) / 4, pos[idx][1].astype(np.float32) / 4
        # map the coordinates
        X = np.arange(y - window_range, y + window_range + 1)
        Y = np.arange(x - window_range, x + window_range + 1)
        indexes = np.transpose([np.tile(Y, len(X)), np.repeat(X, len(Y))])
        curr_desc = map_coordinates(im, [indexes[:, 0],
                                         indexes[:, 1]],
                                    order=1, prefilter=False).reshape(K, K)
        # normalize the descriptor
        E = np.mean(curr_desc)
        curr_desc = (curr_desc - E) / np.linalg.norm(curr_desc - E)
        desc[:, :, idx] += curr_desc
    return desc


def im_to_points(im):
    #############################################################
    # implements the function in example_panoramas.py
    #############################################################
    pyr, vec = sut.build_gaussian_pyramid(im, 3, 3)
    return find_features(pyr)


def find_features(pyr):
    #############################################################
    # finds features in an image given by its pyramid pyr
    #############################################################
    pos = sad.spread_out_corners(pyr[0], 7, 7, 12)
    desc = sample_descriptor(pyr[2], pos, 3)
    return pos, desc


# --------------------------3.2-----------------------------#

def match_features(desc1, desc2, min_score):
    #############################################################
    # matches two descriptors taken from desc1, desc2 according
    # to some minimal score min_score
    #############################################################
    match_ind1, match_ind2 = [], []
    desc1_2nd, desc2_2nd = {}, {}

    flat_desc1 = np.array(map(np.ndarray.flatten, np.rollaxis(np.array(desc1), 2)))
    flat_desc2 = np.array(map(np.ndarray.flatten, np.rollaxis(np.array(desc2), 2)))

    for (idx1, d1), (idx2, d2) in product(enumerate(flat_desc1),
                                          enumerate(flat_desc2)):
        # filtering by the conditions
        # condition 1
        dot = np.inner(d1, d2)
        if not dot > min_score:
            continue
        # condition 2
        if idx1 not in desc1_2nd:
            desc1_2nd[idx1] = sorted(np.dot(d1, np.transpose(flat_desc2)))[-2]
        if not dot >= desc1_2nd[idx1]:
            continue
        # condition 3
        if idx2 not in desc2_2nd:
            desc2_2nd[idx2] = sorted(np.dot(d2, np.transpose(flat_desc1)))[-2]
        if not dot >= desc2_2nd[idx2]:
            continue
        # if they fulfill the conditions, they match
        match_ind1.append(idx1)
        match_ind2.append(idx2)

    return match_ind1, match_ind2


# --------------------------3.3-----------------------------#

def apply_homography(pos1, H12):
    #############################################################
    # applying homographic transformation on given indexes
    #############################################################
    expand = np.column_stack((pos1, np.ones(len(pos1))))
    dot = np.dot(H12, expand.T).T
    normalizes = (dot.T / dot[:,2]).T
    return np.delete(normalizes, -1, axis=1)


def ransac_homography(pos1, pos2, num_iters, inlier_tol):
    #############################################################
    # applying RANSAC routine on the matches
    #############################################################
    pos1, pos2 = np.array(pos1), np.array(pos2)
    best_inliers = []
    for i in range(num_iters):
        # extract 4 random point and compute homography
        idx = np.random.random_integers(0, pos1.shape[0] - 1, size=4)
        points1, points2 = pos1[idx], pos2[idx]
        H12 = sad.least_squares_homography(points1, points2)
        # avoid unstable results
        if H12 is None:
            continue
        to_pos2 = np.array(apply_homography(pos1, H12))
        # compute amount of inliers
        in_indices = np.where(np.array(map(np.sum, (to_pos2 - pos2) ** 2)) < inlier_tol)[0]
        best_inliers = in_indices if len(in_indices) > len(best_inliers) else best_inliers
    # recompute the homography
    points1, points2 = pos1[best_inliers], pos2[best_inliers]
    H12 = sad.least_squares_homography(points1, points2)
    return H12, best_inliers


def display_matches(im1, im2, pos1, pos2, inliers):
    #############################################################
    # display the matches detected by RANSAC
    #############################################################
    plt.figure()
    plt.imshow(np.hstack((im1, im2)), 'gray')
    for idx in range(len(pos1)):
        color = 'y' if idx in inliers else 'b'
        plt.plot([pos1[idx, 1], pos2[idx, 1] + im1.shape[1]],
                 [pos1[idx, 0], pos2[idx, 0]], mfc='r', c=color, lw=.4, ms=5, marker='o')
    plt.show()
    return


# --------------------------3.3-----------------------------#

def accumulate_homographies(H_successive, m):
    #############################################################
    # compute accumulated homographies between successive images
    #############################################################
    if not m:
        return [np.eye(3), np.linalg.inv(H_successive[0])]
    left_slice, right_slice = H_successive[:m], map(np.linalg.inv, H_successive[m:])
    left_slice = sut.accumulate(left_slice[::-1], np.dot)[::-1]
    right_slice = sut.accumulate(right_slice, np.dot)
    left_slice.append(np.eye(3))
    H2m = np.array(left_slice + right_slice)
    H2m = (H2m.T / H2m[:,1,1]).T
    return H2m


# --------------------------4.3-----------------------------#

def prepare_panorama_base(ims, Hs):
    #############################################################
    # ims - the list of images, Hs - the list of homographies
    #############################################################
    corner_points = np.zeros((len(ims), 4))
    centers = np.zeros((len(ims), 2))
    for idx in range(len(ims)):
        rows, cols = float(ims[idx].shape[0]), float(ims[idx].shape[1])
        corners = [[0, 0], [0, cols - 1], [rows - 1, 0], [rows - 1, cols - 1]]
        new_corners = np.array(apply_homography(corners, Hs[idx]))

        corner_points[idx, 0] = np.max(new_corners[:, 0])
        corner_points[idx, 1] = max(np.min(new_corners[:, 0]), 0)
        corner_points[idx, 2] = np.max(new_corners[:, 1])
        corner_points[idx, 3] = max(np.min(new_corners[:, 1]), 0)

        # also on the center
        centers[idx] = [(rows-1)/2, (cols-1)/2]
        centers[idx] = np.array(apply_homography([centers[idx]], Hs[idx]))
    return corner_points, centers


def render_panorama(ims, Hs):
    #############################################################
    # rendering the panorama produces by the images ims
    #############################################################

    corners, centers = prepare_panorama_base(ims, Hs)

    Xmin, Xmax = np.min(corners[:, 1]), np.max(corners[:, 2])
    Ymin, Ymax = np.min(corners[:, 3]), np.max(corners[:, 0])

    Ypano, Xpano = np.meshgrid(np.arange(Xmin, Xmax), np.arange(Ymin, Ymax))
    panorama = np.zeros_like(Xpano)
    # calculate borders
    borders = [0]
    for i in range(len(ims)-1):
        borders.append(np.round((centers[i,1]+centers[i+1,1])/2))
    borders.append(panorama.shape[1])
    # rendering
    for i in range(len(ims)):
        left_border, right_border = int(borders[i]), int(borders[i+1])
        X, Y = Xpano[:,left_border:right_border], Ypano[:,left_border:right_border]
        indices = np.array(apply_homography(np.transpose([X.ravel(), Y.ravel()]), np.linalg.inv(Hs[i])))
        panorama[:,left_border:right_border] += \
            map_coordinates(ims[i], [indices[:,0], indices[:,1]], order=1, prefilter=False)\
                .reshape(panorama[:,left_border:right_border].shape)
    return panorama



def main():
    # a = np.array([[2.,3.,6.], [5.,6.,1.], [3.,8.,9.]])
    # print a
    # print map_coordinates(a, [[0,1], [2,0]])
    #
    im = sut.read_image('external/backyard1.jpg', 1)

    H = np.array([[1,1,2], [1,3,1], [9,1,1]])
    a = [[2,3], [4,5], [8,8]]
    b = np.column_stack((a, np.ones(len(a))))
    c = np.dot(H, b.T).T
    d = (c.T / c[:,2]).T
    e = np.delete(d, -1, axis=1)

    n = np.arange(9).reshape(3,3)
    m = np.arange(9).reshape(3,3) + 1
    x = m + 6

    v = np.array([n,m,x])

    # pos = sad.spread_out_corners(im, 7, 7, 40)
    # plt.imshow(im, 'gray')
    # plt.scatter(pos[:,0], pos[:,1])
    # plt.show()
    # pos = harris_corner_detector(im)
    #
    # pyr, vec = sut.build_gaussian_pyramid(im, 3, 3)
    #
    # # for i in pyr:
    # #     plt.imshow(i, 'gray')
    # #     plt.show()
    #
    # desc = sample_descriptor(pyr[2], pos, 3)

    # b = sample_descriptor(a, np.array([[3,3]]), 1)
    # print a; print b[:,:,0];


    # plt.imshow(im, 'gray')
    # for i in range(desc.shape[2]):
    #     print desc[:,:,i].shape
    #     plt.imshow(desc[:,:,i], 'gray')
    #     plt.show()

    window_x_before = 0
    window_x_after = 2
    window_y_before = 0
    window_y_after = 2

    # ccc = np.empty((7,7,400))
    # print ccc[:, :, 0].shape

    x = np.arange(window_x_before, window_x_after + 1).astype(np.float32) / 2
    y = np.arange(window_y_before, window_y_after + 1).astype(np.float32) / 2

    indexes = np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])
    # print indexes
    # print a; print b
    # print np.concatenate((a, b), axis=1)
    # print map_coordinates(a[:,:,0], [indexes[:,1], indexes[:,0]], order=1, prefilter=False).reshape(3,3)

    a = np.array(
        [[[1, 2, 3], [1, 2, 4], [3, 3, 3]], [[1, 2, 3], [1, 2, 4], [3, 3, 3]], [[1, 2, 3], [1, 2, 4], [3, 3, 3]]])
    b = np.array(
        [[[2, 3, 1], [2, 2, 2], [1, 1, 1]], [[2, 3, 1], [2, 2, 2], [1, 1, 1]], [[2, 3, 1], [2, 2, 2], [1, 1, 1]]])
    # print map(np.ndarray.flatten, np.array(a))
    # print map(np.ndarray.flatten, a)
    # for ix, i in enumerate(map(np.ndarray.flatten, a)):
    #     print i
    # i = [[1,2], [3,4], [7,7]]
    # g = h = [1,2,3,4]
    # f = [1,2,3,4,5,6,7,8,9]
    # print f[:6], f[6:]
    # print map(lambda x: reduce(lambda _,y: x*y, g), h)
    #
    # i.insert(len(i), i)
    # print list(sut.accumulate(g, np.dot))


if __name__ == '__main__':
    main()