import numpy as np
#from BP3MG.PSF_tools.conv2D_fourier import conv2D_fourier
from scipy.signal import convolve

def apply_PSFvar3Dz_block(x_share, list_n3, z, a):

    """
    Apply, at depth z, 3D PSF a = h(z), to 3D volume x
    """

    N1, N2 = x_share.shape[:2]
    p3 = int((a.shape[2]-1)/2)
    Hxz = np.zeros((N1,N2))

    #for n3 = max(1, z-p3):min(N3, z + p3)
    for i in range(len(list_n3)):
        #bF2 = conv2D_fourier(x_share[:,:,i], a[:,:,z-list_n3[i]+p3])
        bF2 = convolve(x_share[:,:,i], a[:,:,z-list_n3[i]+p3], 'same')
        #print(np.linalg.norm(x_share[:,:,i].flatten()),np.linalg.norm(a[:,:,z-list_n3[i]+p3]),np.linalg.norm(bF2))
        Hxz = Hxz + bF2

    return Hxz