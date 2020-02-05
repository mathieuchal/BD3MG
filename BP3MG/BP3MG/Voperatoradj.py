
def Vhoperatoradj(x,nx,ny):
    x = x.reshape(nx,ny)
    Vhtx = x.copy()
    Vhtx[:,0:ny-1] = x[:,0:ny-1]-x[:,1:ny]
    return Vhtx.flatten()

def Vvoperatoradj(x,nx,ny):
    x = x.reshape(nx,ny)
    Vvtx = x.copy()
    Vvtx[0:nx-1,:] = x[0:nx-1,:]-x[1:nx,:]
    return Vvtx.flatten()