import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from BP3MG.PSF_tools.apply_PSFvar3D import apply_PSFvar3D
from BP3MG.PSF_tools.apply_PSFadjvar3D import apply_PSFadjvar3D
from BP3MG.PSF_tools.gaussian_kernel_3D import gaussian_kernel_3D
from MajorizeMinimizeMemoryGradient3D import MajorizeMinimizeMemoryGradient3D

Nx = 64
Ny = Nx
Nz = 24
I = loadmat('/home/mathieuchalvidal/PycharmProjects/untitled/Images/FlyBrain.mat')['I']
sli = slice(0, 256, int(256/64))
I = I[sli, sli, :Nz]

eta = 0.1
lambda_ = 1
kappa = 2
delta = 1e-3
phi = [4,1]

Nh = np.array([5,5,11]).astype(int)
Sx = loadmat('/home/mathieuchalvidal/Desktop/Matlab/Sx.mat')['Sx']
Sy = loadmat('/home/mathieuchalvidal/Desktop/Matlab/Sy.mat')['Sy']
Sz = loadmat('/home/mathieuchalvidal/Desktop/Matlab/Sz.mat')['Sz']
#Sx = np.random.rand(Nz, 1) * 3
#Sy = np.random.rand(Nz, 1) * 3
#Sz = np.random.rand(Nz, 1) * 4
Phiy = np.random.rand(Nz, 1) * 2 * np.pi * 0
Phiz = loadmat('/home/mathieuchalvidal/Desktop/Matlab/Phiz.mat')['Phiz']

#getting convolution kernel
h = lambda z : gaussian_kernel_3D(((Nh-1)/2).astype(int), [Sx[z], Sy[z], Sz[z]], [Phiy[z], Phiz[z]])
Hop = lambda x : apply_PSFvar3D(x, h)
Hop_adj = lambda y : apply_PSFadjvar3D(y, h)

y = Hop(I)

test =  MajorizeMinimizeMemoryGradient3D(y,Hop,Hop_adj,eta, lambda_,kappa, delta, phi ,np.zeros((64,64,24)),I,I,0,1,10000,600)
SNRend, SNR, result, Crit, Ndx, Time, Mem  = test.optimize()

"""
for eta in [0.1,1,10]:
    for lambda_kappa_delta in [0.00001,0.001,1,10]:

        test =  MajorizeMinimizeMemoryGradient3D(convolve(brain, G, 'same'),G,G.T,eta, lambda_kappa_delta, [1,1],np.ones((256,256,57)),brain,brain,0,1,100,1000)
        SNRend, SNR, result, Crit, Ndx, Time, Mem = test.optimize()

        name = 'eta={}, lambda, delta, kappa  = {}, phi = [1,1], zeros'.format(eta,lambda_kappa_delta)
        plt.imshow(result[:,:,30])
        plt.title(name)
        plt.savefig(name+'.png')
        plt.close()
        with open(name+'.txt', 'w') as file:
            file.write(' '.join(np.array(Crit).astype(str)))
            file.write('\n')
            file.write(' '.join(np.array(Ndx).astype(str)))
            file.write('\n')
            file.write(' '.join(np.array(Time).astype(str)))
            file.write('\n')
            file.write(' '.join(np.array(Mem)))
            file.close()
"""

plt.plot(loadmat('/home/mathieuchalvidal/Downloads/bp3mg/Critold.mat')['Critold'].T,label='matlab')
plt.plot(Crit,label='python')
plt.legend()
plt.show()