import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh


np.set_printoptions(linewidth=2000,precision=4, suppress=True)

# configure
N = 8  # number of antennas
M = 5   # number of beams
kA = 3  # number of signals which are equal.
k_true = np.array([0.3, 0.4,0.5,0.6,0.7])  # beam angles
SNR = 5  # 信噪比(dB)

# create a manifold vector
def steering_vector(k, N):
    n = np.arange(N)
    return np.exp(1j * 2 * np.pi * k * n)

NSamples = 100

# generate signals
X = np.zeros((N, NSamples), dtype=complex)
for i in range(M):
    a_k = steering_vector(k_true[i], N)
    if i >= kA:
        s = np.exp(1j * 2 * np.pi * np.random.rand(NSamples))  # random signals
    X += np.outer(a_k, s)

# add noise
noise = (np.random.randn(N, NSamples) + 1j * np.random.randn(N, NSamples)) / np.sqrt(2)
X += noise * (10 ** (-SNR / 20))

# covariance matrix
R_y = X @ X.conj().T / X.shape[1]

# Eigenvalue Decomposition
eigvals, eigvecs = eigh(R_y)
print(eigvals)
#U_n = eigvecs[:, :-(M-kA+1)]  # noise sub-space
U_n = eigvecs[:, :-M]  # noise sub-space
print(U_n.shape)

# MUSIC pseudo-spectrum
k_scan = np.linspace(0, 1, 500)
P_music = np.zeros_like(k_scan, dtype=float)

for i, k in enumerate(k_scan):
    a_k = steering_vector(k, N)
    P_music[i] = 1 / np.abs(a_k.conj().T @ U_n @ U_n.conj().T @ a_k)

# normalize
P_music = 10 * np.log10(P_music / np.max(P_music))

# 绘制 MUSIC 频谱
plt.plot(k_scan, P_music)
plt.xlabel("wave beam index (normalized) ")
plt.ylabel("MUSIC Spectrum (dB)")
plt.title("MUSIC High-Resolution f-k Spectrum")
plt.grid()
plt.xticks(np.arange(0, 1, 0.1))
plt.show()
