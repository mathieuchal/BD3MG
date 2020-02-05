import numpy as np
from BP3MG.Voperator import Voperator
from BP3MG.apply2PSFadjvar3Dz import apply2PSFadjvar3Dz_block
from BP3MG.Gradient2D import gradient2D
from PSF_tools.conv2D_fourier import conv2D_fourier
from PSF_tools.conv2Dadjoint_fourier import conv2Dadjoint_fourier
from BP3MG.majorantem import majorantem
from scipy.signal import convolve
from scipy.io import loadmat

def LOOP3MGpar(z, init, x_share, list_n3, dxz, Htyz, H1z, H, Nx, Ny, Nz, eta, lambda_, delta, tau, phi, xmin, xmax, p3):

    szmin = max(0,z-p3)
    szmax = min(Nz,z+p3+1)

    locz = np.where(list_n3==z)[0]
    xz = x_share[:,:,locz].flatten()
    
    #gradient of data fidelity
    HtHxz = apply2PSFadjvar3Dz_block(x_share, list_n3, z, H, Nz)
    gradfz = HtHxz - Htyz

    #(Nx*Ny) flat gradient of spatial transversal regularization

    if (z == 0):
        loczp = np.where(list_n3==z+1)[0]
        xzp = x_share[:,:,loczp].flatten()
        dregz = xz-xzp

    elif(z == Nz-1):
        loczm = np.where(list_n3==z-1)[0]
        xzm = x_share[:,:,loczm].flatten()
        dregz = xz-xzm
         
    else:
        loczp = np.where(list_n3==z+1)[0]
        xzp = x_share[:, :,loczp].flatten()
        loczm = np.where(list_n3==z-1)[0]
        xzm = x_share[:, :,loczm].flatten()
        dregz = 2*xz-xzm-xzp

    #complete gradient computation
    Gradz, wVxz = gradient2D(xz, gradfz.flatten(), eta, lambda_ , delta, phi, xmin, xmax, Nx, Ny)
    Gradz = Gradz + tau*dregz

    #computing update given the subspace

    if (init == 1):
        Gradz = Gradz.reshape((-1,1))
        Dirz = -Gradz.copy()  #(Nx*Ny)
        V1D, V2D = Voperator(Dirz, Nx, Ny) #(Nx*Ny) flat gradients
        V1D = V1D.reshape((-1, 1))
        V2D = V2D.reshape((-1, 1))
        dirz = Dirz.reshape(Nx,Ny)
        HtTHDirZ = np.zeros((Nx,Ny))

        for zz in range(szmin,szmax):
            a = H[zz]
            azz = a[:,:,zz-z+p3]
            Ftheta = convolve(np.ones((Nx,Ny)),azz,'same')/H1z
            #Ftheta = conv2D_fourier(np.ones((Nx,Ny)),azz)/H1z
            Fdirz = convolve(dirz,azz,'same')
            #Fdirz =conv2D_fourier(dirz,azz)
            Ftheta[Ftheta==0]=1
            thdirz = Fdirz/Ftheta
            HtTHDirZ = HtTHDirZ + conv2Dadjoint_fourier(thdirz, azz)

        DtAD = np.dot(Dirz.flatten(),HtTHDirZ.flatten())

    else:

        Dirz1 = -Gradz
        dirz1 = Dirz1.reshape((Nx,Ny))
        Dirz2 = dxz.flatten()
        dirz2 = Dirz2.reshape((Nx,Ny))
        Dirz = np.vstack([Dirz1, Dirz2]).T
        V1d1, V2d1 = Voperator(Dirz1, Nx, Ny)  # flat gradients
        V1d2, V2d2 = Voperator(Dirz2, Nx, Ny)  # flat gradients
        V1D = np.vstack([V1d1, V1d2]).T
        V2D = np.vstack([V2d1, V2d2]).T
        HtTHDirZ1 = np.zeros((Nx,Ny))
        HtTHDirZ2 = np.zeros((Nx,Ny))

        for zz in range(szmin,szmax):
            a = H[zz]
            azz = a[:,:,zz-z+p3]
            Ftheta = conv2D_fourier(np.ones((Nx,Ny)),azz)/H1z
            Fdirz1 = conv2D_fourier(dirz1, azz)
            thdirz1 = Fdirz1/Ftheta
            thdirz1[Ftheta == 0] = 0
            Fdirz2 = conv2D_fourier(dirz2,azz)
            thdirz2 = Fdirz2/Ftheta
            thdirz2[Ftheta == 0] = 0
            HtTHDirZ1 = HtTHDirZ1 + conv2Dadjoint_fourier(thdirz1, azz)
            HtTHDirZ2 = HtTHDirZ2 + conv2Dadjoint_fourier(thdirz2, azz)

        DtAD = np.array([np.dot(Dirz1.flatten(),HtTHDirZ1.flatten()), np.dot(Dirz2.flatten(),HtTHDirZ1.flatten()), np.dot(Dirz1.flatten(),HtTHDirZ2.flatten()), np.dot(Dirz2.flatten(),HtTHDirZ2.flatten())]).reshape((2,2))


    Bz = majorantem(xz, wVxz, V1D, V2D, DtAD, Dirz, eta, lambda_ , xmin, xmax)
    #print('BZ',z, Bz.astype(str))

    Bz = Bz + 2*tau*Dirz.T.dot(Dirz)

    sz = -np.linalg.pinv(Bz,rcond=1e-20) @ Dirz.T.dot(Gradz) #rcond=1e-16
    dxz = Dirz @ sz
    xz = xz.reshape((-1,1)) + dxz.reshape((-1,1))

    return xz, dxz
