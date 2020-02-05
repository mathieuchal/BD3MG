import numpy as np

def majorantem(x, wVx, V1D, V2D, DtHtHD, D, eta, lambda_ , xmin, xmax):

    m = V1D.shape[-1]
    epsilon = 1e-30
    wVx = np.tile(wVx,(m,1)).T

    DtVtWVD = V1D.T.dot(wVx*V1D) + V2D.T.dot(wVx*V2D)

    D1 = D[x<=xmin]
    D2 = D[x>=xmax]

    B = DtHtHD + lambda_*DtVtWVD + eta *(D1.T.dot(D1) + D2.T.dot(D2)) + epsilon*np.eye(m)

    return B
