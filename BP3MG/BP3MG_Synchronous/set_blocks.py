import numpy as np

def set_blocks(cores_number, Nz):

    blocklist = dict(zip([i for i in range(cores_number)],[[] for i in range(cores_number)]))
    blocksize = np.floor(Nz/cores_number).astype(int)

    #for j in range(cores_number):
    #    blocklist[j] = np.arange(j*blocksize,(j+1)*blocksize)

    for j in range(Nz):
        blocklist[j%cores_number].append(j)

    #for the last block, add all the remaining indexes
    #blocklist[cores_number-1] = np.arange((cores_number-1)*blocksize,Nz)

    return blocklist