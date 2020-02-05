import numpy as np
from PSF_tools.conv2D_fourier import conv2D_fourier
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

    N3 = x.shape[2]
    Hx=np.zeros(x.shape)

    if (not callable(h)):
        a = h[0]
    else:
        a = h

    p3 = int((a.shape[2]-1)/2)
    sa3 = 2*p3+1

    # Boucle sur le z du stack
    if (not callable(h)):
        for z in range(N3):
            a = h[z]
            for n3 in range(max(0,z-p3),min(N3,sa3+z-1-p3)):
                bF2 = conv2D_fourier(x[:,:,n3],a[:,:,z-n3+p3+1])
                Hx[:,:,z] = Hx[:,:,z] + bF2
    else:
        Hx = conv3D_fourier(x,h)
        #Hx = convolve(x,h, 'same') #use of scipy.signal package

    return Hx
    

