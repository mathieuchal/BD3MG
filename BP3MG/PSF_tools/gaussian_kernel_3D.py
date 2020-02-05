import numpy as np
from PSF_tools.rotation3d import rotation3d

def gaussian_kernel_3D(k, sigma, phi):

    """
    k: rayon --> flou de taille 2 * k + 1
    sigma: ecart - type
    Wirjadi, O.,Breuel, T., "Approximate separable 3D anisotropic Gauss filter," in Image
    Processing, 2005.
    ICIP 2005 IEEE International Conference on, vol .2, no., pp.II - 149 - 52, 11 - 14 Sept. 2005
    doi: 10.1109 / ICIP .2005 .1530013
    URL: http: // ieeexplore.ieee.org / stamp / stamp.jsp?tp = & arnumber = 1530013 & isnumber = 32661
    """

    N1 = 2*k[0] + 1
    N2 = 2*k[1] + 1
    N3 = 2*k[2] + 1
    g = np.zeros((N1, N2, N3))
    X = np.arange(-k[0],k[0]+1)
    Y = np.arange(-k[1],k[1]+1)
    Z = np.arange(-k[2],k[2]+1)
    phiy = phi[0]
    phiz = phi[1]
    for i in range(N1):
        for j in range(N2):
            for l in range(N3):
                u, v, w = rotation3d(X[i], Y[j], Z[l], phiy, phiz)
                g[i, j, l] = np.exp(-0.5*np.sum(np.divide([u**2,v**2,w**2],[sigma[0]**2,sigma[1]**2,sigma[2]**2])))
                #(u**2/(2*sigma[0]**2) + v**2/(2*sigma[1]**2) + w**2/(2*sigma[2]**2)))
    g = np.divide(g,np.sum(g),where=g!=0,out=np.zeros_like(g))

    return g