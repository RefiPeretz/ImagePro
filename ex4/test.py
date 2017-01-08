import sol4_utils as util
import sol4_add as add
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
from itertools import *
# # x = 4
# # y = 6
# # xs = np.meshgrid(np.arange(x-3,x + 4), np.arange(x-3,x+4))[1]
# # ys = np.meshgrid(np.arange(y-3, y + 4), np.arange(y-3, y+4))[0]
# #
# # # print(xs+x)
# # # print(ys+y)
# # print(xs)
# # print(ys)
# #
# # print(np.dstack((xs,ys)).reshape(49,2))
#
#
# a = np.arange(12.).reshape((4, 3))
#
# b = np.array([[0.5, 1, 0.5], [0.5, 2, 0.5]])
# c = b[:,0]
# d = b[:,1]
# e = np.array([c,d])
# print(a)
#
# print(map_coordinates(a, b,order=1))

# papo = np.arange(1,4)
# papo1 = np.arange(1,6)
#
# c = list(product(papo, papo1))
# # print(c)
# papo = [-1,-1]

def calc_s_matrix(desc1, desc2):
    sd1 = desc1.shape
    sd2 = desc2.shape
    d1 = np.transpose(desc1.reshape((sd1[0] * sd1[1], sd1[2])))
    d2 = desc2.reshape((sd2[0] * sd2[1], sd2[2]))
    S = d1.dot(d2)
    return S


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
