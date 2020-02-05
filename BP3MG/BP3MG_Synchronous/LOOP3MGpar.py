import numpy as np
from BP3MG.BP3MG_Synchronous.Voperator import Voperator
from BP3MG.BP3MG_Synchronous.apply2PSFadjvar3Dz import apply2PSFadjvar3Dz_block
from BP3MG.BP3MG_Synchronous.Gradient2D import gradient2D
from BP3MG.PSF_tools.conv2D_fourier import conv2D_fourier
from BP3MG.PSF_tools.conv2Dadjoint_fourier import conv2Dadjoint_fourier
from BP3MG.BP3MG_Synchronous.majorantem import majorantem
from scipy.signal import convolve
from scipy.io import loadmat

import matplotlib.pyplot as plt

def LOOP3MGpar(z, init, x_share, list_n3, dxz, Htyz, H1z, H, Nx, Ny, Nz, eta, lambda_, delta, tau, phi, xmin, xmax, p3):

    szmin = max(0,z-p3)
    szmax = min(Nz,z+p3+1)

    locz = np.where(list_n3==z)[0]
    #np.searchsorted(list_n3, np.arange(zzmin, zzmax + 1))
    xz = x_share[:,:,locz].flatten()
    #gradient of data fidelity
    HtHxz = apply2PSFadjvar3Dz_block(x_share, list_n3, z, H, Nz)
    gradfz = HtHxz - Htyz

    # (Nx*Ny) flat gradient of spatial transversal regularization

    if (z == 0):
        loczp = np.where(list_n3==z+1)[0]
        xzp = x_share[:,:,loczp].flatten()
        dregz = xz-xzp

    elif(z == Nz-1):
        loczm = np.where(list_n3==z-1)[0]
        xzm = x_share[:,:,loczm].flatten()
        dregz = xz-xzm
         
    else:
        loczp = np.where(list_n3==z+1)[0]
        xzp = x_share[:, :,loczp].flatten()
        loczm = np.where(list_n3==z-1)[0]
        xzm = x_share[:, :,loczm].flatten()
        dregz = 2*xz-xzm-xzp

    #complete gradient   # verified
    Gradz, wVxz = gradient2D(xz, gradfz.flatten(), eta, lambda_ , delta, phi, xmin, xmax, Nx, Ny)
    Gradz = Gradz + tau*dregz

    if (init == 1):
        Gradz = Gradz.reshape((-1,1))
        Dirz = -Gradz.copy()  #(Nx*Ny)
        V1D, V2D = Voperator(Dirz, Nx, Ny) #(Nx*Ny) flat gradients
        V1D = V1D.reshape((-1, 1))
        V2D = V2D.reshape((-1, 1))
        dirz = Dirz.reshape(Nx,Ny)
        HtTHDirZ = np.zeros((Nx,Ny))

        for zz in range(szmin,szmax):
            a = H[zz]
            azz = a[:,:,zz-z+p3]
            Ftheta = convolve(np.ones((Nx,Ny)),azz,'same')/H1z
            #Ftheta = conv2D_fourier(np.ones((Nx,Ny)),azz)/H1z
            Fdirz = convolve(dirz,azz,'same')
            #Fdirz =conv2D_fourier(dirz,azz)
            Ftheta[Ftheta==0]=1
            thdirz = Fdirz/Ftheta
            HtTHDirZ = HtTHDirZ + conv2Dadjoint_fourier(thdirz, azz)

        DtAD = np.dot(Dirz.flatten(),HtTHDirZ.flatten())

    else:

        Dirz1 = -Gradz
        dirz1 = Dirz1.reshape((Nx,Ny))
        Dirz2 = dxz.flatten()
        dirz2 = Dirz2.reshape((Nx,Ny))
        Dirz = np.vstack([Dirz1, Dirz2]).T
        V1d1, V2d1 = Voperator(Dirz1, Nx, Ny)  # flat gradients
        V1d2, V2d2 = Voperator(Dirz2, Nx, Ny)  # flat gradients
        V1D = np.vstack([V1d1, V1d2]).T
        V2D = np.vstack([V2d1, V2d2]).T
        HtTHDirZ1 = np.zeros((Nx,Ny))
        HtTHDirZ2 = np.zeros((Nx,Ny))

        for zz in range(szmin,szmax):
            a = H[zz]
            azz = a[:,:,zz-z+p3]
            Ftheta = conv2D_fourier(np.ones((Nx,Ny)),azz)/H1z
            Fdirz1 = conv2D_fourier(dirz1, azz)
            thdirz1 = Fdirz1/Ftheta
            thdirz1[Ftheta == 0] = 0
            Fdirz2 = conv2D_fourier(dirz2,azz)
            thdirz2 = Fdirz2/Ftheta
            thdirz2[Ftheta == 0] = 0
            HtTHDirZ1 = HtTHDirZ1 + conv2Dadjoint_fourier(thdirz1, azz)
            HtTHDirZ2 = HtTHDirZ2 + conv2Dadjoint_fourier(thdirz2, azz)

        DtAD = np.array([np.dot(Dirz1.flatten(),HtTHDirZ1.flatten()), np.dot(Dirz2.flatten(),HtTHDirZ1.flatten()), np.dot(Dirz1.flatten(),HtTHDirZ2.flatten()), np.dot(Dirz2.flatten(),HtTHDirZ2.flatten())]).reshape((2,2))

    #print('DtAD', DtAD.astype(str))
    #print('HtHxz', np.linalg.norm(HtHxz,1))
    #print(z,list_n3)
    #print([np.linalg.norm(x_share[:,:,k].flatten(),1).astype(str) for k in range(x_share.shape[2])])
    #print(np.linalg.norm(dxz.flatten(), 1).astype(str) )

    Bz = majorantem(xz, wVxz, V1D, V2D, DtAD, Dirz, eta, lambda_ , xmin, xmax)
    #print('BZ',z, Bz.astype(str))

    Bz = Bz + 2*tau*Dirz.T.dot(Dirz)

    sz = -np.linalg.pinv(Bz,rcond=1e-20) @ Dirz.T.dot(Gradz) #rcond=1e-16
    #print('pinv(Bz)',z,np.linalg.pinv(Bz).astype(str))
    dxz = Dirz @ sz
    xz = xz.reshape((-1,1)) + dxz.reshape((-1,1))

    return xz, dxz#, Bz, DtAD, HtTHDirZ, Ftheta,thdirz, dirz, Gradz, HtHxz

"""
import numpy as np
import multiprocessing as mp
from scipy.io import loadmat
from BP3MG.PSF_tools.gaussian_kernel_3D import gaussian_kernel_3D
from BP3MG.blur_alt_z import blur_alt_z
from BP3MG.adjblur_alt_z import adjblur_alt_z
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

#getting convolution kernel
def h(z):
    return gaussian_kernel_3D(((Nh-1)/2).astype(int), [Sx[z], Sy[z], Sz[z]], [Phiy[z], Phiz[z]])
print('size kernel: Nx = {}, Ny = {}, Nz = {}'.format(*h(0).shape))

#SNR = 20;
sigma = 0.02

#add blur and noise in a parrallel fashion
pool = mp.Pool(mp.cpu_count())
results = [pool.apply(blur_alt_z, args=(I, Nh, Nx, Ny, Sx, Sy, Sz, Phiy, Phiz, sigma, z)) for z in range(Nz)]
pool.close()
Iblurnoisy,  Iblurz, BSNRinitz, SNRinitz  = zip(*results)
Iblurnoisy = np.dstack(Iblurnoisy)
Iblurz = np.dstack(Iblurz)
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

print('done')


#Regularization parameters:
lambda_ = 1
delta = 2
phi = 4
#Bounds of the constrained domain:
xmin = 0
xmax = 1
#Elastic net parameter:
tau = 1e-3
#Weight of the quadratic distance function :Gradz
eta = 0.1
#Initialization
x0 = np.zeros((Nx, Ny, Nz))


Hty_2 = loadmat('/Users/GrandesTerres1/PycharmProjects/untitled/Seeds/Hty.mat')['Hty'].T

z=11
list_n3 = np.arange(max(0,z-2*5),min(Nz,z+2*5))
print('Hty = ', Hty.reshape(4096,24))
print('Hty original=', Hty)


xloc,dxloc = LOOP3MGpar(z, 1, Iblurnoisy[:,:,z-2*5:z+2*5+1], list_n3, np.zeros((Nx,Ny)), Hty_2.reshape(64,64,24)[:,:,z], H1[:,:,z], h, Nx, Ny, Nz, eta, lambda_, delta, tau, phi, xmin, xmax, 5)
plt.imshow(xloc.reshape(64,64))
print(xloc[-10:])
plt.show()
"""