import numpy as np
#from PSF_tools.gaussian_kernel_3D import gaussian_kernel_3D
from BP3MG.PSF_tools.applyPSFadjvar3Dz import applyPSFadjvar3Dz
from scipy.signal import convolve
from BP3MG.h import h

def adjblur_alt_z(Iblurnoisy, z, Nh, Nx, Ny, Nz, Sx, Sy, Sz, Phiy, Phiz):
    Htyz = applyPSFadjvar3Dz(Iblurnoisy, z, Nh, Sx, Sy, Sz, Phiy, Phiz)
    H1Z = convolve(np.ones((Nx, Ny, Nz)), h(z, Nh, Sx, Sy, Sz, Phiy, Phiz),'same')
    return Htyz, H1Z

"""

import time
import multiprocessing as mp
from scipy.io import loadmat
from BP3MG.PSF_tools.gaussian_kernel_3D import gaussian_kernel_3D
from BP3MG.blur_alt_z import blur_alt_z
import matplotlib.pyplot as plt

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


#SNR = 20;
sigma = 0.02

#add blur and noise in a parrallel fashion
start = time.time()
pool = mp.Pool(mp.cpu_count())
results = [pool.apply(blur_alt_z, args=(I, Nh, Nx, Ny, Sx, Sy, Sz, Phiy, Phiz, sigma, z)) for z in range(Nz)]
pool.close()

Iblurnoisy,  Iblurz, BSNRinitz, SNRinitz  = zip(*results)
Iblurnoisy = np.dstack(Iblurnoisy)
Iblurz = np.dstack(Iblurz)

cpu_time_blur = time.time()-start

SNRinit = 10*np.log10(np.sum(I**2)/np.sum((I-Iblurnoisy)**2))
BSNRinit = 10*np.log10(np.sum(I**2)/np.sum((I-Iblurz)**2))
print('SNR init = ', str(SNRinit),', BSNRinit = ', str(BSNRinit))

Iblurnoisy = loadmat('/Users/GrandesTerres1/PycharmProjects/untitled/Seeds/Iblurnoisy.mat')['Iblurnoisy']

y = Iblurnoisy.reshape(Nx*Ny, Nz)
#we need these vectors in the algorithm

pool = mp.Pool(mp.cpu_count())
results = [pool.apply(adjblur_alt_z, args=(Iblurnoisy, z, Nh, Nx, Ny, Nz, Sx, Sy, Sz, Phiy, Phiz)) for z in range(Nz)]
pool.close()

Hty, H1Z  = zip(*results)
Hty= np.dstack(Hty)
H1 = np.dstack([H1Z[z][:,:,z] for z in range(len(H1Z))])

Hty_ = loadmat('/Users/GrandesTerres1/PycharmProjects/untitled/Seeds/Hty.mat')['Hty'].reshape(64,64,24)
Hty_ = np.swapaxes(Hty_,0,1)

print('Hty diff',Hty-Hty_)
plt.imshow((Hty-Hty_)[:,:,20])
plt.colorbar()
plt.show()
"""