import numpy as np
from photutils.datasets import make_4gaussians_image
from photutils.centroids import (centroid_1dg, centroid_2dg,
                                 centroid_com, centroid_quadratic)
import matplotlib.pyplot as plt


data = make_4gaussians_image()
plt.imshow(data, cmap='Blues', origin='lower')#, vmin=median_s-2*std_s, vmax=median_s+5*std_s)
data -= np.median(data[0:30, 0:125])
plt.figure()
plt.imshow(data, cmap='Blues', origin='lower')#, vmin=median_s-2*std_s, vmax=median_s+5*std_s)

#x1, y1 = centroid_com(data)
#print(np.array((x1, y1)))
#x2, y2 = centroid_quadratic(data)
#print(np.array((x2, y2)))
#x3, y3 = centroid_1dg(data)
#print(np.array((x3, y3)))
x4, y4 = centroid_2dg(data)
print(np.array((x4, y4)))
#plt.plot(x1, y1, 'rx', label='COM')
#plt.plot(x2, y2, 'gx', label='Quadratic')
#plt.plot(x3, y3, 'yx', label='1D Gaussian')
plt.plot(x4, y4, 'mx', label='2D Gaussian')
plt.legend()

plt.show()

