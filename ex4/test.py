import sol4_utils as util
import sol4_add as add
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
# x = 4
# y = 6
# xs = np.meshgrid(np.arange(x-3,x + 4), np.arange(x-3,x+4))[1]
# ys = np.meshgrid(np.arange(y-3, y + 4), np.arange(y-3, y+4))[0]
#
# # print(xs+x)
# # print(ys+y)
# print(xs)
# print(ys)
#
# print(np.dstack((xs,ys)).reshape(49,2))


a = np.arange(12.).reshape((4, 3))

b = np.array([[0.5, 1, 0.5], [0.5, 2, 0.5]])
c = b[:,0]
d = b[:,1]
e = np.array([c,d])
print(a)

print(map_coordinates(a, b,order=1))
