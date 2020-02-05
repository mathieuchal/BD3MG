import numpy as np
from scipy.signal import convolve

def Gaussian3D(sigma_array, size_array):

    if (np.all(sigma_array > 0)) and (len(size_array) == len(sigma_array)):

        sigma_x = sigma_array[0]
        size_x = size_array[0]
        sigma_y = sigma_array[1]
        size_y = size_array[1]
        sigma_z = sigma_array[2]
        size_z = size_array[2]

        x = np.arange(-np.ceil(size_x/2),np.ceil(size_x/2)+1)
        Kx = np.exp(-(x**2/(2*(sigma_x**2))))
        Kx = Kx/np.sum(Kx)

        y = np.arange(-np.ceil(size_y/2),np.ceil(size_y/2)+1)
        Ky = np.exp(-(y**2/(2*(sigma_y**2))))
        Ky = Ky/np.sum(Ky)

        z = np.arange(-np.ceil(size_z/2),np.ceil(size_z/2)+1)
        Kz = np.exp(-(z**2/(2*(sigma_z**2))))
        Kz = Kz/np.sum(Kz)

        Hx = Kx.reshape(len(Kx),1,1)
        Hy = Ky.reshape(1,len(Ky),1)
        Hz = Kz.reshape(1,1,len(Kz))

        #since gaussian Kernel is separable
        G = convolve(Hz, convolve(Hx, Hy))

    return G
