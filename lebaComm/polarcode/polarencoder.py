
import numpy as np

"""
    极化码的结构是没有做置换操作的，即编码矩阵是  G = F^n
"""

# Polar encode x without using recursion
# x = message bit field
# it seems that using G matrix is not faster than not-using G.
def polarEncode(x,N,frozen=None,G=None):
    """
    Encodes a message using polar coding with a non-recursive implementation.
    The message x is encoded using in-place operations of output u.
    This function is copied and modified from https://github.com/mcba1n/polar-codes.
    """

    # fill frozen bits and data bits x
    
    if frozen is not None:
        u = np.zeros(N,dtype=int)
        lut = np.ones(N, dtype=int)
        lut[frozen] = 0
        u[lut==1] = x
    else:
        u = x.copy()

    # loop over the M stages
    #if G is None:
    n = int(np.log2(N))
    nn = N
    if G is None:
        for i in range(n):
            if nn == 1:  # base case: when partition length is 1
                break
            n_split = int(nn / 2)

            # select the first index for each split partition, we always have n=2*n_split
            for p in range(0, N, nn):
                # loop through left partitions and add the right partitions for a previous partition
                for k in range(n_split):
                    l = p + k
                    u[l] = u[l] ^ u[l + n_split]
            nn = n_split
    else:
        u = u @ G
        u = np.mod(u,2)
    
    return u

# to create the encoder matrix G.
#     G = F^n
def polarCreateMatrixG(N):
    n = int(np.log2(N))
    G = np.array([[1,0],[1,1]])
    F = np.array([[1,0],[1,1]])
    for i in range(n-1):
        G = np.kron(G,F)
    
    return G
    


