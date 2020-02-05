import numpy as np
from PSF_tools.conv2D_fourier import conv2D_fourier
def apply_PSFvar3Dz(x, z, a):

    """
    Apply, at depth z, 3D PSF a = h(z), to 3D volume x
    """
    N1, N2, N3 = x.shape
    Hxz = np.zeros((N1, N2))

    p3 = int((a.shape[2]-1)/2)

    zmin = max(0,z-p3)
    zmax = min(N3,z+p3+1)

    for n3 in range(zmin,zmax):
            bF2 = conv2D_fourier(x[:,:,n3], a[:,:,z-n3+p3])
            Hxz = Hxz + bF2

    return Hxz