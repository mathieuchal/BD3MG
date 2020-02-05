import numpy as np
from PSF_tools.conv3D_fourier import conv3D_fourier
#from scipy.signal import convolve


"""
def apply_PSFvar3D(x,h):

    N0, N1, N2 = x.shape
    Hx = np.zeros(x.shape)
    for z in range(N2):
        Hx[:,:,z] =convolve(x, h(z),'same')[:,:,z]

    return Hx
"""

def apply_PSFvar3D(x,h):

    N1,N2,N3 = x.shape
    Hx=np.zeros(x.shape)

    # Boucle sur le z du stack
    #if (callable(h)):
    for z in range(N3):
        a = h(z)
        bF2 = conv3D_fourier(x,a)
        Hx[:,:,z] = bF2[:,:,z]
    #else:
    #    Hx = conv3D_fourier(x,h)

    return Hx
