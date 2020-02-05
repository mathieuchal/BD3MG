def Voperator3D(x):
    
    nx, ny, nz = x.shape
    
    Vvx = x.copy()
    Vvx[1:nx,:,:] = x[1:nx,:,:]-x[0:nx-1,:,:]
    
    Vhx = x.copy()
    Vhx[:,1:ny,:] = x[:,1:ny,:]-x[:,0:ny-1,:]
    
    Vtx = x.copy()
    Vtx[:,:,1:nz] = x[:,:,1:nz]-x[:,:,0:nz-1]
    
    return Vvx, Vhx, Vtx

def Vhoperatoradj3D(x):
    ny = x.shape[1]
    Vhtx = x.copy()
    Vhtx[:,0:ny-1,:] = x[:,0:ny-1,:]-x[:,1:ny,:]

    return Vhtx

def Vtoperatoradj3D(x):
    nz = x.shape[2]
    Vttx = x.copy()
    Vttx[:,:,0:nz-1] = x[:,:,0:nz-1]-x[:,:,1:nz]

    return Vttx

def Vvoperatoradj3D(x):
    nx = x.shape[1]
    Vvtx = x.copy()
    Vvtx[0:nx-1,:,:] = x[0:nx-1,:,:]-x[1:nx,:,:]

    return Vvtx
