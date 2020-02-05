def Voperator(x, nx, ny):
    x = x.reshape(nx,ny)
    Vvx=x.copy()
    Vvx[1:nx,:] = x[1:nx,:]-x[0:nx-1,:]
    Vvx = Vvx.flatten()

    Vhx=x.copy()
    Vhx[:,1:ny] = x[:,1:ny]-x[:,0:ny-1]
    Vhx = Vhx.flatten()

    return Vvx, Vhx
