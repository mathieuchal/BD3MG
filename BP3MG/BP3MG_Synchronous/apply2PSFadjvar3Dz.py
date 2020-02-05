from PSF_tools.conv2Dadjoint_fourier import conv2Dadjoint_fourier
from PSF_tools.apply_PSFvar3Dz_block import apply_PSFvar3Dz_block
import numpy as np

def apply2PSFadjvar3Dz_block(x_share, list_n3, z, H, N3):

    """
    Ht(H(x))for varying PSF h (handle) and x, result at depth z
    """

    N1, N2 = x_share.shape[:2]
    HtHxz = np.zeros((N1,N2))
    p3 = int((H[0].shape[2]-1)/2)
    zmin = max(0,z-p3)
    zmax = min(N3,z+p3+1)

    for n3 in range(zmin,zmax):
        a = H[n3]
        #HX = conv3D_fourier(x, a);
        #Hxn3 = HX(:,:, z);
        zzmin = max(0,n3-p3)
        zzmax = min(N3,n3+p3+1)
        #Hxn3 = apply_PSFvar3Dz(x, n3, a);
        locz = np.searchsorted(list_n3, np.arange(zzmin,zzmax))
        #print('boucle n3',n3)
        Hxn3 = apply_PSFvar3Dz_block(x_share[:,:,locz],np.arange(zzmin,zzmax), n3, a)
        bF2 = conv2Dadjoint_fourier(Hxn3, a[:,:,n3-z+p3])
        #print(np.linalg.norm(Hxn3,2), np.linalg.norm(bF2,2))
        HtHxz = HtHxz + bF2

    return HtHxz
