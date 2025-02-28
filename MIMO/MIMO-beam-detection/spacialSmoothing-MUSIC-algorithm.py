import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd

np.set_printoptions(linewidth=2000,precision=4, suppress=True)

# Generate coherent signals (highly correlated signals)
def generate_coherent_signals(M, N, angles, wavelength=1.0, d=0.5):
    """
    Generate N sampled coherent signals received by an M-element array.
    :param M: Number of array elements
    :param N: Number of samples
    :param angles: List of incident signal angles (degrees)
    :param wavelength: Signal wavelength
    :param d: Element spacing (typically 0.5 * wavelength)
    :return: X (M × N matrix) - Array received signals
    """
    k = 2 * np.pi / wavelength  # Wave number
    theta_rad = np.radians(angles)  # Convert angles to radians
    A = np.exp(1j * k * d * np.outer(np.arange(M), np.sin(theta_rad)))  # Array manifold matrix
    S = np.random.randn(len(angles), N) + 1j * np.random.randn(len(angles), N)  # Complex random signals
    S[1, :] = S[0, :]  # Make the second signal coherent with the first one
    X = A @ S  # Received signals
    noise = (np.random.randn(M, N) + 1j * np.random.randn(M, N)) * 0.1  # Add noise
    return X + noise

# Compute covariance matrix
def compute_covariance(X):
    return X @ X.conj().T / X.shape[1]

# Forward spatial smoothing
def spatial_smoothing(R, L):
    """
    Apply spatial smoothing to the covariance matrix R.
    :param R: Original covariance matrix
    :param L: Subarray size
    :return: Smoothed covariance matrix
    """
    M = R.shape[0]  # Total number of elements
    smoothed_R = np.zeros((L, L), dtype=complex)
    num_subarrays = M - L + 1  # Number of subarrays
    for i in range(num_subarrays):
        smoothed_R += R[i:i+L, i:i+L]
    return smoothed_R / num_subarrays  # Compute the average

# MUSIC algorithm
def music_algorithm(R, M, num_sources, scan_angles):
    """
    MUSIC angle estimation
    :param R: Covariance matrix
    :param M: Number of array elements
    :param num_sources: Estimated number of sources
    :param scan_angles: Range of search angles
    :return: Estimated angles
    """
    U, Lambda, Vh = svd(R)
    noise_subspace = Vh[num_sources:].conj().T  # Noise subspace
    angles_rad = np.radians(scan_angles)
    steering_matrix = np.exp(1j * 2 * np.pi * 0.5 * np.outer(np.arange(M), np.sin(angles_rad)))
    P_music = 1 / np.linalg.norm(steering_matrix.conj().T @ noise_subspace, axis=1)
    return scan_angles, 10 * np.log10(P_music / np.max(P_music))  # Normalize and convert to dB

# Parameters
M = 8  # Number of elements
N = 200  # Number of samples
angles = [10, 30, 60]  # There signal angles, two being coherent
num_sources = len(angles)  # Estimated number of sources
scan_angles = np.linspace(-90, 90, 360)  # Search angles
L = 5  # Subarray size for spatial smoothing

# Generate signals and compute covariance matrices
X = generate_coherent_signals(M, N, angles)
R_original = compute_covariance(X)  # Original covariance matrix
R_smoothed = spatial_smoothing(R_original, L)  # Smoothed covariance matrix

# MUSIC angle estimation
angles_raw, spectrum_raw = music_algorithm(R_original, M, num_sources, scan_angles)
angles_smooth, spectrum_smooth = music_algorithm(R_smoothed, L, num_sources, scan_angles)

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(angles_raw, spectrum_raw, label="Original MUSIC", linestyle="dashed", color="red")
plt.plot(angles_smooth, spectrum_smooth, label="Spatial Smoothed MUSIC", color="blue")
plt.xlabel("Angle (degrees)")
plt.ylabel("Spatial Spectrum (dB)")
plt.legend()
plt.title("MUSIC Algorithm with and without Spatial Smoothing")
plt.xticks(np.arange(-90, 90, 10))
plt.grid()
plt.show()
