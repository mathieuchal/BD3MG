from numpy.fft import ifftn,fftn
import numpy as np
from PSF_tools.conv2Dadjoint_fourier import conv2Dadjoint_fourier

def apply_PSFadjvar3D(x, h):

    N0, N1, N2 = x.shape
    Hstarx = np.zeros(x.shape)


    if (callable(h)):
        a = h(1)
    else:
        a = h

    p3 = int((a.shape[2]-1)/2)
    sa3 = 2*p3+1

    if (not callable(h)):

        p = np.floor(np.array(a.shape)/2).astype(int)
        Dext = np.zeros(np.array(x.shape+2*p,'int'))
        Dext[p[0]:p[0]+N0, p[1]:p[1]+N1, p[2]:p[2]+N2]= x

        H = np.conj(freqfilt3D(a,2*p[0]+N0,2*p[1]+N1,2*p[2]+N2))

        Hstarx = ifftn(fftn(Dext)*H)
        Hstarx = Hstarx[p[0]:p[0]+N0, p[1]:p[1]+N1, p[2]:p[2]+N2]
        print(np.linalg.norm(Hstarx.flatten(),1))
    else:
        # Boucle sur le z du stack
        for z in range(N2):
            for n3 in range(max(0,z-p3),min(N2,sa3+z-p3)):
                a = h(n3)
                bF2 = conv2Dadjoint_fourier(x[:,:,n3],a[:,:,n3-z+p3])
                Hstarx[:,:,z] = Hstarx[:,:,z] + bF2

    return Hstarx

