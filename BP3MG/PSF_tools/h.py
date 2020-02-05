from PSF_tools.gaussian_kernel_3D import gaussian_kernel_3D

def h(z, Nh, Sx, Sy, Sz, Phiy, Phiz):
    return gaussian_kernel_3D(((Nh-1)/2).astype(int), [Sx[z], Sy[z], Sz[z]], [Phiy[z],Phiz[z]])