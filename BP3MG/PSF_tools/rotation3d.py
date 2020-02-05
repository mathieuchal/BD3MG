import numpy as np

def rotation3d(x, y, z, phiy, phiz):

    u = np.cos(phiy)*np.cos(phiz)*x + np.sin(phiz)*y - np.sin(phiy)*np.cos(phiz)*z
    v = -np.cos(phiy)*np.sin(phiz)*x + np.cos(phiz)*y + np.sin(phiy)*np.sin(phiz)*z
    w = np.sin(phiy)*x + np.cos(phiy)*z

    return u, v, w