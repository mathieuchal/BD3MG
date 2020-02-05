
def Vvoperatoradj3D(x):
    nx = x.shape[1]
    Vvtx = x
    Vvtx[0:nx-1,:,:] = x[0:nx-1,:,:]-x[1:nx,:,:]

    return Vvtx
