import numpy as np
from polarcode.utils import *

"""
    极化码的结构是没有做置换操作的，即编码矩阵是  G = F^n
"""

def polarConBhatBound(N, K, design_SNR_normalised):
    """
    Polar code construction using Bhattacharyya Bounds. Each bit-channel can have different parameters.
    Supports shortening by adding extra cases for infinite likelihoods.

    This function is copied and modified from https://github.com/mcba1n/polar-codes.
    
    **References:**

    * Vangala, H., Viterbo, E., & Hong, Y. (2015). A Comparative Study of Polar Code Constructions for the AWGN Channel. arXiv.org. Retrieved from http://search.proquest.com/docview/2081709282/

    """

    n = int(np.log2(N))
    z = np.zeros((N, n + 1))
    z0 = -design_SNR_normalised
    z[:, 0] = z0  # initial channel states

    for j in range(1, n + 1):
        u = 2 ** j  # number of branches at depth j
        for t in range(0, N, u):  # loop over top branches at this stage
            for s in range(int(u / 2)):
                k = t + s
                z_top = z[k, j - 1]
                z_bottom = z[k + int(u / 2), j - 1]

                if z_top != z_bottom:
                    print(z_top,z_bottom)
                # shortening infinity cases
                if z_top == -np.inf and z_bottom != -np.inf:
                    z[k, j] = z_bottom
                    z[k + int(u / 2), j] = -np.inf
                elif z_top != -np.inf and z_bottom == -np.inf:
                    z[k, j] = z_top
                    z[k + int(u / 2), j] = -np.inf
                elif z_top == -np.inf and z_bottom == -np.inf:
                    z[k, j] = -np.inf
                    z[k + int(u / 2), j] = -np.inf
                # principal equations
                else:
                    z[k, j] = logdomain_diff(logdomain_sum(z_top, z_bottom), z_top + z_bottom)
                    z[k + int(u / 2), j] = z_top + z_bottom

    reliabilities = np.argsort(-z[:, n], kind='mergesort')   # ordered by least reliable to most reliable
    frozen = np.argsort(z[:, n], kind='mergesort')[K:]     # select N-K least reliable channels
    FERest = FER_estimate(frozen, z[:, n])
    z = z[:, n]
    return reliabilities, frozen, FERest

def perfect_pcc(N, K, p):
    """
    Boolean expression approach to puncturing pattern construction.

    This function is copied and modified from https://github.com/mcba1n/polar-codes.

    **References:**

    * Song-Nam, H., & Hui, D. (2018). On the Analysis of Puncturing for Finite-Length Polar Codes: Boolean Function Approach. arXiv.org. Retrieved from http://search.proquest.com/docview/2071252269/

    """

    n = int(np.log2(N))
    z = np.zeros((myPC.N, n + 1), dtype=int)
    z[:, 0] = p

    for j in range(1, n + 1):
        u = 2 ** j  # number of branches at depth j
        # loop over top branches at this stage
        for t in range(0, N, u):
            for s in range(int(u / 2)):
                k = t + s
                z_top = z[k, j - 1]
                z_bottom = z[k + int(u / 2), j - 1]
                z[k, j] = z_top & z_bottom
                z[k + int(u / 2), j] = z_top | z_bottom
    return z[:, n]

def general_ga(N, K, design_SNR_normalised):
    """
    Polar code construction using density evolution with the Gaussian Approximation. Each channel can have different parameters.

    This function is copied and modified from https://github.com/mcba1n/polar-codes.

    **References:**

    * Trifonov, P. (2012). Efficient Design and Decoding of Polar Codes. IEEE Transactions on Communications, 60(11), 3221–3227. https://doi.org/10.1109/TCOMM.2012.081512.110872

    * Vangala, H., Viterbo, E., & Hong, Y. (2015). A Comparative Study of Polar Code Constructions for the AWGN Channel. arXiv.org. Retrieved from http://search.proquest.com/docview/2081709282/

    """

    n = int(np.log2(N))
    z = np.zeros((N, n + 1))

    z0 = np.array([4 * design_SNR_normalised] * N)
    z[:, 0] = z0  # initial channel states

    for j in range(1, n + 1):
        u = 2 ** j  # number of branches at depth j
        # loop over top branches at this stage
        for t in range(0, N, u):
            for s in range(int(u / 2)):
                k = t + s
                z_top = z[k, j - 1]
                z_bottom = z[k + int(u / 2), j - 1]

                z[k, j] = phi_inv(1 - (1 - phi(z_top)) * (1 - phi(z_bottom)))
                z[k + int(u / 2), j] = z_top + z_bottom

    m = np.array([logQ_Borjesson(0.707*np.sqrt(z[i, n])) for i in range(N)])
    reliabilities = np.argsort(-m, kind='mergesort')    # ordered by least reliable to most reliable
    frozen = np.argsort(m, kind='mergesort')[K:]     # select N-K least reliable channels
    FERest = FER_estimate(frozen, m)
    z = m
    return reliabilities, frozen, FERest

def FER_estimate(frozen, z):
    # This function is copied and modified from https://github.com/mcba1n/polar-codes.
    
    FERest = 0
    for i in range(len(z)):
        if i not in frozen:
            FERest = FERest + np.exp(z[i]) - np.exp(z[i]) * FERest
    return FERest

