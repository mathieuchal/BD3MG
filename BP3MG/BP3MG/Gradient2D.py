import numpy as np
from BP3MG.Voperator import Voperator
from BP3MG.Voperatoradj import Vhoperatoradj, Vvoperatoradj

def gradient2D(x, dJ, eta, lambda_ , delta, phi, xmin, xmax, Nx, Ny):

    Vvx, Vhx = Voperator(x, Nx, Ny)  #(Nx*Ny) flat gradient
    Vx = np.sqrt(Vvx**2 + Vhx**2)    #(Nx*Ny) flat gradient

    if phi == 1:
        wVx = (1/(delta**2))*np.exp(-(Vx**2)/(2*delta**2))
    elif phi == 2:
        wVx = (4*delta**2)/(2*delta**2 + Vx**2)**2
    elif phi == 3:
        wVx = 2/(delta**2 + Vx**2)
    elif phi == 4:
        wVx = (1/(delta**2))*(1+(Vx**2)/delta**2)**(-1/2)
    elif phi == 5:
        wVx = np.ones(np.prod(Vx.shape[:2]))

    dF = dJ + lambda_*(Vhoperatoradj(Vhx*wVx, Nx, Ny) + Vvoperatoradj(Vvx*wVx, Nx, Ny))

    if np.isfinite(xmin):
        dF = dF + eta*np.where(x<=xmin, x-xmin, 0)

    if np.isfinite(xmax):
        dF = dF + eta*np.where(x>=xmax, x-xmax ,0)

    return dF, wVx