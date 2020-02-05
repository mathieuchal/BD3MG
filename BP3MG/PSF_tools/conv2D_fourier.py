import numpy as np
from numpy.fft import fft2, ifft2
from BP3MG.PSF_tools.freqfilt2D import freqfilt2D

def conv2D_fourier(D, a):
    """
    function bF2 = conv2D_fourier(D, a)
    input
    D = 2D data
    a = 2D PSF
    output
    bF2 = a * D(stationary convolution with zero padding), same size as D.
        """


    p = np.floor(np.array(a.shape)/2).astype(int)
    N1, N2 = D.shape

    Dext = np.zeros(D.shape+2*p)
    Dext[p[0]:N1+p[0], p[1]:p[1]+N2] = D

    A = freqfilt2D(a,2*p[0]+N1,2*p[1]+N2)

    bF2 = ifft2(fft2(Dext)*A).real
    bF2 = bF2[p[0]:N1+p[0],p[1]:p[1]+N2]

    return bF2