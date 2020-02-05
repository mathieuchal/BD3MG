
def Vhoperatoradj3D(x):
    ny = x.shape[1]
    Vhtx = x
    Vhtx[:,0:ny-1,:] = x[:,0:ny-1,:]-x[:,1:ny,:]

    return Vhtx
