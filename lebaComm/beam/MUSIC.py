import numpy as np
from scipy.linalg import eigh


# This function is to calculate the spectrum by giving the actual beam number.
# Parameters:
#     Y: received signal for each antenna
#     N: number of antennas
#     M: no of actural beams
#     NPoints: number of the beams in which to draw the spectrum.
#  Output:
#     k_scan: points where to calculate the specturm power
#     P_music:  spectrum power
def MusicGetSpectrum(Y,N,M,NPoints):
    # covariance matrix
    R_y = Y @ Y.conj().T / Y.shape[1]

    # Eigenvalue Decomposition
    eigvals, eigvecs = eigh(R_y)
    U_n = eigvecs[:, :-M]  # noise sub-space

    # MUSIC pseudo-spectrum
    k_scan = np.linspace(0, 1, NPoints)
    P_music = np.zeros_like(k_scan, dtype=float)

    for i, k in enumerate(k_scan):
        a_k = steering_vector(k, N)
        P_music[i] = 1 / np.abs(a_k.conj().T @ U_n @ U_n.conj().T @ a_k)

    # normalize
    P_music = 10 * np.log10(P_music / np.max(P_music))

    return k_scan,P_music

# create a manifold vector
def steering_vector(k, N):
    n = np.arange(N)
    return np.exp(1j * 2 * np.pi * k * n)


def rootMUSIC(Y, num_sources, d=0.5, wavelength=1.0):
    """
    Root-MUSIC DOA estimation (for ULA).
    
    Parameters:
        Y: the received signals (num_sensors x num_snapshots)
        num_sources: number of signals (number of DOAs to be estimated)
        d: sensor spacing (in wavelengths, default 0.5)
        wavelength: signal wavelength (default 1.0)
    
    Returns:
        doa_estimates_deg: estimated DOAs (in degrees, sorted in ascending order)
    """
    num_snapshots = Y.shape[1]
    R = Y @ Y.conj().T / num_snapshots

    num_sensors = R.shape[0]
    
    # Perform eigenvalue decomposition of the covariance matrix
    eigvals, eigvecs = eigh(R)
    
    # Select the noise subspace: use the (num_sensors - num_sources) eigenvectors corresponding to the smallest eigenvalues
    En = eigvecs[:, :num_sensors - num_sources]
    
    # Construct the noise subspace projection matrix
    Pn = En @ En.conj().T  # Size: (num_sensors x num_sensors)
    
    # Extract polynomial coefficients using the Toeplitz structure:
    # For a ULA, each diagonal of Pn should theoretically be equal,
    # Here, sum the elements on each diagonal to obtain the coefficients c[k] (k from -M+1 to M-1)
    c = np.array([np.sum(np.diag(Pn, k)) for k in range(-num_sensors+1, num_sensors)])
    
    # Normalize: set the coefficient for k=0 (the main diagonal) to 1, which does not change the root locations
    c = c / c[num_sensors - 1]
    
    # Construct the polynomial coefficients; note that np.roots requires the coefficients in descending order
    poly_coeffs = c[::-1]
    
    # Solve for all roots of the polynomial
    roots_all = np.roots(poly_coeffs)
    
    # Consider only the roots inside the unit circle (theoretically, the signal-related roots should lie near the unit circle)
    roots_inside = roots_all[np.abs(roots_all) < 1]
    
    # Sort the roots by their distance from the unit circle and select the num_sources roots closest to the unit circle
    distances = np.abs(np.abs(roots_inside) - 1)
    sorted_indices = np.argsort(distances)
    selected_roots = roots_inside[sorted_indices][:num_sources]
    
    # According to theory, the phase of the root satisfies: angle(z) = -2π*d*sin(θ)/wavelength
    # beta = 2π*d/wavelength
    beta = 2 * np.pi * d / wavelength
    phi = np.angle(selected_roots)
    
    doa_estimates_rad = np.arcsin(phi / beta)
    doa_estimates_deg = np.rad2deg(doa_estimates_rad)
    
    return np.sort(doa_estimates_deg),roots_all

