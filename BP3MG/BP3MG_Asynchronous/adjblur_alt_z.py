import numpy as np
#from PSF_tools.gaussian_kernel_3D import gaussian_kernel_3D
from PSF_tools.applyPSFadjvar3Dz import applyPSFadjvar3Dz
from scipy.signal import convolve
from PSF_tools.h import h

def adjblur_alt_z(Iblurnoisy, z, Nh, Nx, Ny, Nz, Sx, Sy, Sz, Phiy, Phiz):
    Htyz = applyPSFadjvar3Dz(Iblurnoisy, z, Nh, Sx, Sy, Sz, Phiy, Phiz)
    H1Z = convolve(np.ones((Nx, Ny, Nz)), h(z, Nh, Sx, Sy, Sz, Phiy, Phiz),'same')
    return Htyz, H1Z
