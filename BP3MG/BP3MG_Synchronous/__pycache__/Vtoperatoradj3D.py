
def Vtoperatoradj3D(x):
    nz = x.shape[2]
    Vttx = x
    Vttx[:,:,0:nz-1] = x[:,:,0:nz-1]-x[:,:,1:nz]

    return Vttx
