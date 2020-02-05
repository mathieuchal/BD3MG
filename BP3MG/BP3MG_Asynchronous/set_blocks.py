import numpy as np

def set_blocks(cores_number, Nz):

    blocklist = dict(zip([i for i in range(cores_number)],[[] for i in range(cores_number)]))
    blocksize = np.floor(Nz/cores_number).astype(int)

    for j in range(Nz):
        blocklist[j%cores_number].append(j)

    return blocklist