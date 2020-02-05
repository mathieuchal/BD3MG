import numpy as np
from PSF_tools.freqfilt3D import freqfilt3D


def conv3D_fourier(D, a):

    """
    # input :
    # D = 3D data
    # a = 3D PSF kernel
    # output :
    # bF2 = a * D(stationary convolution with zero padding), same size as D.
    """

    p = np.floor(np.array(a.shape)/2).astype(int)
    N1, N2, N3 = D.shape

    Dext = np.zeros(D.shape + 2*p)
    Dext[p[0]:p[0]+N1, p[1]:p[1]+N2, p[2]:p[2]+N3] = D

    A = freqfilt3D(a, 2*p[0]+N1, 2*p[1]+N2, 2*p[2]+N3)

    bF2 = np.fft.ifftn(np.fft.fftn(Dext)*A).real
    bF2 = bF2[p[0]:p[0]+N1, p[1]:p[1]+N2, p[2]:p[2]+N3]

    return bF2