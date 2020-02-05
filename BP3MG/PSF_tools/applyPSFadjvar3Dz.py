from PSF_tools.conv2Dadjoint_fourier import conv2Dadjoint_fourier
import numpy as np
from PSF_tools.h import h

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

