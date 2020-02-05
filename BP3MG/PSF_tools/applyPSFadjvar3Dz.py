from BP3MG.PSF_tools.conv2Dadjoint_fourier import conv2Dadjoint_fourier
import numpy as np
from BP3MG.h import h

def applyPSFadjvar3Dz(x, z, Nh, Sx, Sy, Sz, Phiy, Phiz):

    """
    % Adjoint convolution between varying PSF h(handle) and x, at depth z
    """

    N1, N2, N3= x.shape
    Htxz = np.zeros((N1,N2))

    p3 = int((h(0, Nh, Sx, Sy, Sz, Phiy, Phiz).shape[2]-1)/2)
    for n3 in range(max(0,z-p3),min(N3,p3+z+1)):
        a = h(n3, Nh, Sx, Sy, Sz, Phiy, Phiz)
        bF2 = conv2Dadjoint_fourier(x[:,:,n3], a[:,:,n3-z+p3])
        Htxz = Htxz + bF2

    return Htxz

"""
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

Htyz = applyPSFadjvar3Dz(I,0,np.array([5,5,11]),Sx,Sy,Sz,Phiy,Phiz)
print(Htyz)
import matplotlib.pyplot as plt

plt.imshow(Htyz)
plt.colorbar()
plt.show()
print(np.max(Htyz))
"""