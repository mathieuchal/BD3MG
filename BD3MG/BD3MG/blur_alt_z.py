import numpy as np
from PSF_tools.gaussian_kernel_3D import gaussian_kernel_3D
from PSF_tools.apply_PSFvar3Dz import apply_PSFvar3Dz


def blur_alt_z(I, Nh, Nx, Ny, Sx, Sy, Sz, Phiy, Phiz, sigma, z):
    h_z = gaussian_kernel_3D(((Nh-1)/2).astype(int), [Sx[z], Sy[z], Sz[z]], [Phiy[z],Phiz[z]])
    Iblurz = apply_PSFvar3Dz(I,z,h_z)
    # sigma = np.std(Iblurz[:])*10**(-SNR/20)
    Iblurnoisyz = Iblurz + np.random.normal(0,sigma,(Nx,Ny))
    Iz = I[:,:,z]
    BSNRinitz = 10*np.log10(np.sum(Iz**2)/np.sum((Iz-Iblurz)**2))
    SNRinitz = 10*np.log10(sum(Iz**2)/np.sum((Iz-Iblurnoisyz)**2))
    return Iblurnoisyz, Iblurz, BSNRinitz, SNRinitz
