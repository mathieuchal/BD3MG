from numpy.fft import fft2
import numpy as np

def freqfilt2D(t, N, M, st=0):

    """
    T = freqfilt2D(t, N, M, st)
    """

    L1, L2 = t.shape

    t = (1+st*(2*np.random.uniform(0,1,(L1,L2))-1))*t
    T = np.zeros((N,M))
    L12 = np.ceil(L1/2).astype(int)
    L22 = np.ceil(L2/2).astype(int)

    T[0:L12,0:L22] = t[L12-1:L1,L22-1:L2]
    T[N-L12+1:N,0:L22] = t[0:L12-1,L22-1:L2]
    T[0:L12,M-L22+1:M] = t[L12-1:L1,0:L22-1]
    T[N-L12+1:N,M-L22+1:M] = t[0:L12-1,0:L22-1]

    T = fft2(T)

    return T

"""
from BP3MG.h import h
import numpy as np
from scipy.io import loadmat
from BP3MG.PSF_tools.gaussian_kernel_3D import gaussian_kernel_3D

Nx = 64
Ny = Nx
Nz = 24
#I = loadmat('/home/mathieuchalvidal/PycharmProjects/untitled/BP3MG/FlyBrain.mat')['I']
I = loadmat('/Users/GrandesTerres1/PycharmProjects/untitled/Images/FlyBrain.mat')['I']
sli = slice(0,256,int(256/64))
I = I[sli,sli,:Nz]
print('Create blurry and noisy image')
print('size image: Nx = ', Nx, ', Ny = ', Ny, ', Nz = ',Nz)

#Degradation parameters
Nh = np.array([5,5,11]).astype(int)
Sx = loadmat('/Users/GrandesTerres1/PycharmProjects/untitled/Seeds/Sx.mat')['Sx']
Sx[0]=2.444171059179537
Sy = loadmat('/Users/GrandesTerres1/PycharmProjects/untitled/Seeds/Sy.mat')['Sy']
Sz = loadmat('/Users/GrandesTerres1/PycharmProjects/untitled/Seeds/Sz.mat')['Sz']
#Sx = np.random.rand(Nz, 1) * 3
#Sy = np.random.rand(Nz, 1) * 3
#Sz = np.random.rand(Nz, 1) * 4
Phiy = np.random.rand(Nz, 1) * 2 * np.pi * 0
Phiz = loadmat('/Users/GrandesTerres1/PycharmProjects/untitled/Seeds/Phiz.mat')['Phiz']
#Phiz = np.random.rand(Nz, 1) * 2 * np.pi

print(freqfilt2D(h(0, Nh, Sx, Sy, Sz, Phiy, Phiz)[:,:,4],68,68))
"""