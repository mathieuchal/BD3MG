import numpy as np
from numpy.fft import fft2, ifft2
from PSF_tools.freqfilt2D import freqfilt2D

def conv2Dadjoint_fourier(D, a):

    """
    % function bF2 = conv2Dadjoint_fourier(D, a)
    % input
    % D = 2D data
    % a = 2D PSF
    % output
    % bF2 = a(-.)*D(adjoint of the stationary convolution with zero padding), same size as D
    """

    p = np.floor(np.array(a.shape)/2).astype(int)
    N1, N2 = D.shape

    Dext = np.zeros(D.shape+2*p)
    Dext[p[0]:p[0]+N1,p[1]:p[1]+N2] = D

    A = freqfilt2D(a,2*p[0]+N1, 2*p[1]+N2).conj()
    #print('A-A_',A-A_)
    #bF2 = ifftn(fftn(D,s=(2*p[0]+N1,2*p[1]+N2))*np.conj(fftn(a,s=(2*p[0]+N1,2*p[1]+N2)))).real
    bF2 = ifft2((fft2(Dext)*A)).real
    bF2 = bF2[p[0]:p[0]+N1,p[1]:p[1]+N2]

    return bF2


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
Sx[0] = 2.444171059179537
Sy = loadmat('/Users/GrandesTerres1/PycharmProjects/untitled/Seeds/Sy.mat')['Sy']
Sz = loadmat('/Users/GrandesTerres1/PycharmProjects/untitled/Seeds/Sz.mat')['Sz']
#Sx = np.random.rand(Nz, 1) * 3
#Sy = np.random.rand(Nz, 1) * 3
#Sz = np.random.rand(Nz, 1) * 4
Phiy = np.random.rand(Nz, 1) * 2 * np.pi * 0
Phiz = loadmat('/Users/GrandesTerres1/PycharmProjects/untitled/Seeds/Phiz.mat')['Phiz']
#Phiz = np.random.rand(Nz, 1) * 2 * np.pi


A = loadmat('/Users/GrandesTerres1/PycharmProjects/untitled/Seeds/A.mat')['A']
fft2_D = loadmat('/Users/GrandesTerres1/PycharmProjects/untitled/Seeds/fft2_D.mat')['fft2_D']

z=10
a = conv2Dadjoint_fourier(I[:,:,z],h(z, Nh, Sx, Sy, Sz, Phiy, Phiz)[:,:,5])
print(a)
import matplotlib.pyplot as plt
plt.imshow(a)
plt.colorbar()
plt.show()
print(np.max(a))
"""