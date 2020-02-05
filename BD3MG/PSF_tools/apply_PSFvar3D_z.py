import numpy as np
from scipy.signal import convolve

def apply_PSFvar3D_z(xz, zx, h, N3):

    """
    Apply PSF on x, such that x(:,:,i) = 0, except for x(: ,:, z) = xz
    """

    Hx = np.zeros(xz.shape[0], xz.shape[1], N3)
    p3 = int((h(1).shape[2]-1)/2)

    zmin = max(1, zx-p3)
    zmax = min(N3,zx+p3)

    # Boucle sur le z du stack
    for z in range(zmin,zmax):
        a = h(z)
        bF2 = convolve(xz, a[:,:,z-zx+p3+1], 'same')
        Hx[:,:, z] = bF2

    return Hx