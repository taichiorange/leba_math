
import numpy as np
import math
from polarcode.utils import *
from polarcode.decoder_utils import *
from polarcode.polarencoder import polarEncode
"""
    极化码的结构是没有做置换操作的，即编码矩阵是  G = F^n
    The structure of the polar code does not include any permutation operation, i.e., the encoding matrix is G = F^n
"""
# Successive Cancellation Decoder (SCD)
def polarScDecode(likelihoods,N,frozen):
    """
    Successive Cancellation Decoder. 
    The decoder will use the frozen set.
    Depends on update_llrs and update_bits.

    This function is copied and modified from https://github.com/mcba1n/polar-codes.
    
    **References:**

    *  Vangala, H., Viterbo, & Yi Hong. (2014). Permuted successive cancellation decoder for polar codes. 
       2014 International Symposium on Information Theory and Its Applications, 438–442. IEICE.

    """

    n = int(np.log2(N))

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)

    L[:, 0] = likelihoods

    # decode bits in natural order
    for l in [bit_reversed(i, n) for i in range(N)]:
        # evaluate tree of LLRs for root index i
        update_llrs(N,n,l,B,L)

        # make hard decision at output
        if l in frozen:
            B[l, n] = 0
        else:
            B[l, n] = hard_decision(L[l, n])

        # propagate the hard decision just made
        update_bits(N,n,l,B)

    u = B[:, n].astype(int)

    # return the decoded bits excluding the frozen bits.
    frozen_lookup = np.ones(N, dtype=int)
    frozen_lookup[frozen] = 0
    return u[frozen_lookup == 1]

# ------------------ for Belief Propagation ------------------
def box_plus(alpha, beta):
    """
    BoxPlus 运算，用于 BP 算法中的 LLR 更新。
    BoxPlus operation, used for LLR updates in the BP algorithm.
    """
    if alpha < beta:
        zmin, zmax = alpha, beta
    else:
        zmin, zmax = beta, alpha
    z = zmax + math.log(1 + math.exp(zmin - zmax))
    z = z - math.log(1.0 + math.exp(alpha + beta))
    return -z
class DataNode:
    """
    数据节点，存储左右传递的软信息以及硬判决值。
    Data node, stores the soft information passed left and right, as well as the hard decision value.
    """
    def __init__(self, L2R=0.0, R2L=0.0):
        self.L2R = L2R
        self.R2L = R2L

# order: determine the BP graph structure.
def polarBpDecode(likelihoodsRatio,N,frozen,maxIter=20,order=None):
    noOfLayers = int(np.log2(N))
    noOfNodesOneLayer = N

    if order is None:
        order = [1<<i for i in range(noOfLayers)]

    SMALLPROB = 1e-9  # 小概率常量，可根据实际情况调整 # Small probability constant, adjustable based on actual needs
    inf = math.log(1 - SMALLPROB) - math.log(SMALLPROB)

    hd_layer0 = np.zeros(N, dtype=int)  # 最左层（layer 0）的硬判决值  # Hard decision values from the leftmost layer (layer 0)
    hd_layern = np.zeros(N, dtype=int)  # 最右层（layer noOfLayers）的硬判决值  # Hard decision values from the rightmost layer (layer noOfLayers)

    # 初始化 data_graph（[层][节点]）——每个节点为一个 DataNode 实例
    # Initialize data_graph ([layer][node]) - each node is an instance of DataNode
    data_graph = [[DataNode() for _ in range(noOfNodesOneLayer)] for _ in range(noOfLayers+1)]
   
    # 初始化最右层（叶节点）的右传信息 R2L 为信道 LLR 值
    # Initialize the right-to-left information R2L of the rightmost layer (leaf nodes) to the channel LLR values
    for index in range(noOfNodesOneLayer):
        data_graph[noOfLayers][index].R2L = likelihoodsRatio[index]

    # 对于冻结位，将最左层（layer 0）的左传信息 L2R 设为 inf
    # For frozen bits, set the left-to-right information L2R of the leftmost layer (layer 0) to inf
    for index in frozen:
        data_graph[0][index].L2R = inf

    #  A -------------- + -------------- B
    #                   |
    #                   |
    #                   |
    #                   |
    #  C -------------------------------- D 


    # 迭代开始：左右信息交替更新  
    # # Iteration begins: left and right messages are updated alternately
    for iter in range(maxIter):
        # 右-->左传播
        # # Right-to-left propagation
        for layer in range(noOfLayers, 0, -1):
            num_in_group = order[layer-1] * 2
            noOfGroup = noOfNodesOneLayer // num_in_group
            for groupId in range(noOfGroup):
                for index in range(num_in_group // 2):
                    A = groupId * num_in_group + index
                    B = A
                    C = groupId * num_in_group + index + (num_in_group // 2)
                    D = C
                    #  A <------------- + <------------- B
                    #                   ^
                    #                   |
                    #                   |
                    #                   |
                    #                   |
                    #  C --------------> <--------------- D 
                    alpha = data_graph[layer][B].R2L
                    beta = data_graph[layer][D].R2L + data_graph[layer-1][C].L2R
                    temp_value = box_plus(alpha, beta)
                    if temp_value > inf:
                        temp_value = inf
                    elif temp_value < -inf:
                        temp_value = -inf
                    data_graph[layer-1][A].R2L = temp_value

                    #  A -------------> + <------------- B
                    #                   |
                    #                   |
                    #                   |
                    #                   |
                    #                   v
                    #  C <-------------- <--------------- D
                    alpha = data_graph[layer][B].R2L
                    beta = data_graph[layer-1][A].L2R
                    temp_value = box_plus(alpha, beta)
                    temp_value += data_graph[layer][D].R2L
                    if temp_value > inf:
                        temp_value = inf
                    elif temp_value < -inf:
                        temp_value = -inf
                    data_graph[layer-1][C].R2L = temp_value

        # 左-->右传播
        # # Left-to-right propagation
        for layer in range(0, noOfLayers):
            num_in_group = order[layer] * 2
            noOfGroup = noOfNodesOneLayer // num_in_group
            for groupId in range(noOfGroup):
                for index in range(num_in_group // 2):
                    A = groupId * num_in_group + index
                    B = A
                    C = groupId * num_in_group + index + (num_in_group // 2)
                    D = C
                    #  A -------------> + -------------> B
                    #                   ^
                    #                   |
                    #                   |
                    #                   |
                    #                   |
                    #  C --------------> <--------------- D 
                    alpha = data_graph[layer][A].L2R
                    beta = data_graph[layer][C].L2R + data_graph[layer+1][D].R2L
                    temp_value = box_plus(alpha, beta)
                    if temp_value > inf:
                        temp_value = inf
                    elif temp_value < -inf:
                        temp_value = -inf
                    data_graph[layer+1][B].L2R = temp_value
                    #  A -------------> + <------------- B
                    #                   |
                    #                   |
                    #                   |
                    #                   |
                    #                   v
                    #  C --------------> ---------------> D
                    alpha = data_graph[layer][A].L2R
                    beta = data_graph[layer+1][B].R2L
                    temp_value = box_plus(alpha, beta)
                    temp_value += data_graph[layer][C].L2R
                    if temp_value > inf:
                        temp_value = inf
                    elif temp_value < -inf:
                        temp_value = -inf
                    data_graph[layer+1][D].L2R = temp_value

        # 硬判决：根据最左层（layer 0）和最右层（layer noOfLayers）的 L2R 和 R2L 值进行硬判决
        # Hard decision: make hard decisions based on the L2R and R2L values from the leftmost layer (layer 0) and the rightmost layer (layer noOfLayers)
        for index in range(N):
            soft_info = data_graph[0][index].L2R + data_graph[0][index].R2L
            hd_layer0[index] = 0 if soft_info >= 0 else 1

        # 硬判决：计算最右层（layer noOfLayers）的判决结果
        # Hard decision: calculate the decision results of the rightmost layer (layer noOfLayers)
        for index in range(N):
            soft_info = data_graph[noOfLayers][index].L2R + data_graph[noOfLayers][index].R2L
            hd_layern[index] = 0 if soft_info >= 0 else 1

        # 提前终止判决：
        # Early termination of decision:
        # 对 hd_layer0 进行一次编码，与 hd_layern 比较，若相同则认为 BP 收敛
        # Perform a encoding on hd_layer0 and compare it with hd_layern. If they are the same, consider BP converged.
        IsSuccess = True
        hd_temp = hd_layer0.copy()
        hd_temp = polarEncode(hd_temp, N)
        for index in range(N):
            if hd_temp[index] != hd_layern[index]:
                IsSuccess = False
                break
        if IsSuccess:
            break

    u = hd_layer0.astype(int)

    # return the decoded bits excluding the frozen bits.
    frozen_lookup = np.ones(N, dtype=int)
    frozen_lookup[frozen] = 0
    return u[frozen_lookup == 1]
