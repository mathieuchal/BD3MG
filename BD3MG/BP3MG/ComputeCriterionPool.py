from PSF_tools.apply_PSFvar3Dz import apply_PSFvar3Dz
from BP3MG.Voperator import Voperator
import numpy as np

def ComputeCriterionPool(z, x, h, y, phi, eta, lambda_, delta, tau, xmin, xmax, Nx, Ny, Nz):

    Hxz = apply_PSFvar3Dz(x.reshape(Nx,Ny,Nz), z, h(z))
    critfz = np.sum((Hxz.flatten()-y[:,z])**2)*0.5
    xz = x[:,:,z]

    Vvx, Vhx = Voperator(xz, Nx, Ny)
    Vx = np.sqrt(Vvx**2 + Vhx**2)

    if phi == 1:
        phiVx = 1-np.exp(-(Vx**2)/(2*delta**2))
    elif phi == 2:
        phiVx = (Vx**2)/(2*delta**2 + Vx**2)
    elif phi == 3:
        phiVx = np.log(1 + (Vx**2)/(delta**2))
    elif phi == 4:
        phiVx = (1+(Vx**2)/(delta**2))**(1/2)-1
    elif phi == 5:
        phiVx = 1/2*Vx**2

    Fz = critfz + lambda_*np.sum(phiVx)

    if np.isfinite(xmin):
        Fz = Fz + eta*np.sum(1/2*(xz[xz<xmin]-xmin)**2)

    if np.isfinite(xmax):
        Fz = Fz + eta*np.sum(1/2* (xz[xz>xmax]-xmax)**2)

    if z == 0:
        xzm = np.zeros((Nx,Ny))
    else:
        xzm = x[:,:,z-1]

    regz = 1/2*np.sum((xz-xzm)**2)
    critz = Fz + tau * regz

    return critz