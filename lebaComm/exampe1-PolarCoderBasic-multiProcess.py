#!/usr/bin/env python3
# 这个是用了多进程的版本，并行执行不同 snr 下的计算，提高速度
# 由于 python 中的多线程并不能并行执行，因此，必须要创建不同的进程来做并行执行。

import numpy as np
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool, cpu_count

from modemod.modem import pskmod
from modemod.modem import pskdemod
from channel.awgn import awgn
from polarcode.polarconstruct import polarConBhatBound
from polarcode.polarencoder import polarEncode
from polarcode.polarencoder import polarCreateMatrixG
from polarcode.polardecoder import polarScDecode

def processTask(task_id,N,K,frozen,snr,NSample):
    pNoise = np.power(10,-snr/10)

    errN = 0
    for j in range(1,NSample):
        # create message data
        message = np.random.randint(2, size=K)

        # polar encode
        xEncoded = polarEncode(message,N,frozen)
        xMod = pskmod(xEncoded)
        txData = awgn(xMod,pNoise)
        pcLlr = pskdemod(txData,pNoise,M=2,OutputType="llr")

        messageReceived = polarScDecode(pcLlr,N,frozen)
        errN += np.sum(messageReceived != message)

    ber = errN/(NSample*K)

    return [snr,errN,ber]

if __name__ == "__main__":
    n = 9
    N = 2**n
    K = 232

    # mothercode construction
    design_SNR  = 5.0
    NSample = int(1e+5)
    realSnr = np.arange(-2,15,1)

    # construct polar code according to the design SNR.
    reliabilities, frozen, FERest = polarConBhatBound(N,K,(10**(design_SNR/10))*K/N)
    print(frozen)

    # G matrix, only demo, no use.
    G = polarCreateMatrixG(N)

    task = []
    for i in range(len(realSnr)):
        snr = realSnr[i]
        task.append((i+1,N,K,frozen,snr,NSample))

    num_cores = cpu_count()
    if num_cores > 4:
        num_cores -= 2
    start_time = time.time()  # Record the start time
    # 使用 Pool 分发任务
    with Pool(processes=num_cores) as pool:
        # 使用 starmap 分发任务（支持多参数函数）
        results = pool.starmap(processTask, task)
    end_time = time.time()  # Record the end time
    print(f"Execution time: {end_time - start_time} seconds")

    results = np.array(results)
    plt.semilogy(results[:,0], results[:,2])
    plt.ylim(1e-6, 0.6)
    plt.grid()
    plt.show(block=True)
