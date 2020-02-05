import numpy as np
from scipy.signal import convolve
from BP3MG.BP3MG_Standard.Gaussian3D import Gaussian3D
from BP3MG.BP3MG_Standard.Voperator3D import *
import scipy.io
import time

def Critere3D(x, y, H, H_adj, eta, lambda_, kappa, delta, phi, xmin, xmax):
    Hxy = H(x) - y
    phiXY = phi[0]
    phiZ = phi[1]

    Vvx, Vhx, Vtx = Voperator3D(x)

    #print(np.linalg.norm(Vvx.flatten(),1))
    #print(np.linalg.norm(Vhx.flatten(),1))
    #print(np.linalg.norm(Vtx.flatten(),1))

    if phiXY == 1:
        phiXY_Vx = (1-np.exp(-((np.sqrt(Vvx**2 + Vhx**2))**2)/(2*delta**2)))
        wXY_Vx = (1/delta**2)*np.exp(-((np.sqrt(Vvx**2 + Vhx**2))**2)/(2*delta**2))
    elif phiXY == 2:
        phiXY_Vx = ((np.sqrt(Vvx**2 + Vhx**2))**2)/(2*delta**2 + (np.sqrt(Vvx**2 + Vhx**2))**2)
        wXY_Vx = (4*delta**2)/(2*delta**2 + (np.sqrt(Vvx**2 + Vhx**2))**2)**2
    elif phiXY == 3:
        phiXY_Vx = np.log(1+((np.sqrt(Vvx**2 + Vhx**2))**2)/(delta**2))
        wXY_Vx = 2/(delta**2 + (np.sqrt(Vvx**2 + Vhx**2))**2)
    elif phiXY == 4:
        phiXY_Vx = (1+((np.sqrt(Vvx**2 + Vhx**2))**2)/(delta**2))**(1/2)-1
        wXY_Vx = (1/(delta**2))*(1+((np.sqrt(Vvx**2 + Vhx**2))**2)/delta**2)**(-1 / 2)
    elif phiXY == 5:
        phiXY_Vx = 1/2*(np.sqrt(Vvx**2 + Vhx**2))**2
        wXY_Vx = np.ones((np.sqrt(Vvx**2 + Vhx**2).shape))
    elif phiXY == 6:
        phiXY_Vx = 1-np.exp(-((1+((np.sqrt(Vvx**2+Vhx**2))**2)/(delta**2))**(1/2)-1))
        wXY_Vx = (1/(delta**2))*(1+((np.sqrt(Vvx**2+Vhx**2))**2)/delta**2)**(-1/2)*np.exp(-((1+((np.sqrt(Vvx**2+Vhx**2))**2)/(delta**2))**(1/2)-1))
    elif phiXY == 7:
        pow = 0.25
        phiXY_Vx = (1+((np.sqrt(Vvx**2 + Vhx**2))**2)/(delta**2))**(pow)-1
        wXY_Vx = ((2*pow)/(delta**2))*(1+((np.sqrt(Vvx**2 + Vhx**2))**2)/delta**2)**(pow - 1)

    p = 0.1
    if phiZ == 1:
        phiZ_Vx = (1-np.exp(-(Vtx**2)/(2*(p)**2)))
        wZ_Vx = (1/((p**2)))*np.exp(-(Vtx**2)/(2*(p)**2))
    elif phiZ == 2:
        phiZ_Vx = (Vtx**2)/(2*(p)**2 + Vtx**2)
        wZ_Vx = (4*(p)**2)/(2*(p)**2 + Vtx**2)**2
    elif phiZ == 3:
        phiZ_Vx = np.log(1+(Vtx**2)/((p)**2))
        wZ_Vx = 2/((p)**2 + Vtx**2)
    elif phiZ == 4:
        phiZ_Vx = (1+(Vtx**2)/(p**2))**(1/2) - 1
        wZ_Vx = (1/(p)**2)*(1+(Vtx**2)/(p)**2)**(-1/2)
    elif phiZ == 5:
        phiZ_Vx = 1/2*Vtx**2
        wZ_Vx = np.ones(Vtx.shape)

    dphiXY_Vvx = Vvx*wXY_Vx
    dphiXY_Vhx = Vhx*wXY_Vx
    dphiZ_Vtx = Vtx*wZ_Vx

    F = 1/2*np.sum(Hxy**2) + lambda_*np.sum(phiXY_Vx) + kappa*np.sum(phiZ_Vx)
    dF = H_adj(Hxy) + lambda_*(Vvoperatoradj3D(dphiXY_Vvx) + Vhoperatoradj3D(dphiXY_Vhx))+ kappa*Vtoperatoradj3D(dphiZ_Vtx)

    if xmin != -np.inf:
        F = F + eta*1/2*np.linalg.norm((x[x<xmin]-xmin))**2
        dF = dF + eta*np.where(x<=xmin,x-xmin,0)

    if xmax != np.inf:
        F = F + eta*1/2*np.linalg.norm((x[x>xmax]-xmax))**2
        dF = dF + eta*np.where((x>=xmax),x-xmax,0)

    return F, dF