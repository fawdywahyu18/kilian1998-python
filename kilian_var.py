import numpy as np
import pandas as pd

# ============================================================================
# Konversi Python dari Kilian (1997) MATLAB code
# olsvarc.m, asybc.m, irfvar.m, boot.m, point.m
# ============================================================================

def olsvarc(y, p):
    """
    Estimasi VAR(p) dengan intercept dalam companion form via OLS.
    y: (T x q)
    Returns: A (companion), SIGMA, U (residuals), V (intercept)
    """
    T, q = y.shape
    y = y.T  # (q x T)

    Y = y[:, p-1:T]  # (q x T-p+1)
    for i in range(1, p):
        Y = np.vstack([Y, y[:, p-1-i:T-i]])  # (q*p x T-p+1)

    X = np.vstack([np.ones((1, T-p)), Y[:, :T-p]])   # (q*p+1 x T-p)
    Y = Y[:, 1:T-p+1]                                 # (q*p x T-p)

    A_full = Y @ X.T @ np.linalg.inv(X @ X.T)
    U      = Y - A_full @ X
    SIGMA  = U @ U.T / (T - p - p*q - 1)
    V      = A_full[:, 0]
    A      = A_full[:, 1:q*p+1]

    return A, SIGMA, U, V


def vec(M):
    """Vectorize matrix column-major (q^2 x 1)."""
    return M.flatten(order='F').reshape(-1, 1)


def asybc(A, SIGMA, T, p):
    """
    Bias correction Pope (1990).
    A: companion matrix (q*p x q*p)
    """
    q_p   = A.shape[0]
    q     = SIGMA.shape[0]
    T_eff = T - p

    vecSIGMAY = np.linalg.inv(np.eye(q_p**2) - np.kron(A, A)) @ vec(SIGMA)
    SIGMAY    = vecSIGMAY.reshape(q_p, q_p, order='F')

    I      = np.eye(q_p)
    B      = A.T
    peigen = np.linalg.eigvals(A)

    sumeig = np.zeros((q_p, q_p), dtype=complex)
    for ev in peigen:
        sumeig += ev * np.linalg.inv(I - ev * B)
    sumeig = sumeig.real

    bias  = SIGMA @ (np.linalg.inv(I - B) + B @ np.linalg.inv(I - B@B) + sumeig) @ np.linalg.inv(SIGMAY)
    Abias = -bias / T_eff

    bcstab = 9.0
    delta  = 1.0

    while bcstab >= 1:
        bcA   = A - delta * Abias
        bcmod = np.abs(np.linalg.eigvals(bcA))
        if np.any(bcmod >= 1):
            bcstab = 1.0
        else:
            bcstab = 0.0
        delta -= 0.01
        if delta <= 0:
            bcstab = 0.0

    return bcA


def irfvar(A, SIGMA, p, h, q):
    """
    Hitung IRF via Cholesky.
    Returns: IRF (q^2 x h+1)
    """
    J   = np.hstack([np.eye(q), np.zeros((q, q*(p-1)))])
    C   = np.linalg.cholesky(SIGMA)  # lower triangular

    IRF = vec(J @ np.linalg.matrix_power(A, 0) @ J.T @ C)
    for i in range(1, h+1):
        IRF = np.hstack([IRF, vec(J @ np.linalg.matrix_power(A, i) @ J.T @ C)])

    return IRF  # (q^2 x h+1)


def boot(A, U, y, V, p, h, q, nrep=1000, seed=42):
    """
    Bootstrap CI untuk IRF (Kilian 1997).
    Returns: CI (2 x q^2*(h+1)) — baris 0: 5th pctile, baris 1: 95th pctile
    """
    rng     = np.random.default_rng(seed)
    T, _    = y.shape
    y       = y.T  # (q x T)

    Y = y[:, p-1:]
    for i in range(1, p):
        Y = np.vstack([Y, y[:, p-1-i:T-i]])  # (q*p x T-p+1)

    IRFrmat = np.zeros((nrep, q**2 * (h+1)))

    for j in range(nrep):

        # Random initial condition
        pos      = rng.integers(0, T-p+1)
        Yr       = np.zeros((q*p, T-p+1))
        Yr[:, 0] = Y[:, pos]

        # Resample residuals
        idx       = rng.integers(0, T-p, size=T-p)
        Ur        = np.zeros((q*p, T-p+1))
        Ur[:q, 1:] = U[:q, idx]

        # Generate pseudo data
        for i in range(1, T-p+1):
            Yr[:, i] = V + A @ Yr[:, i-1] + Ur[:, i]

        # Reconstruct yr (T x q)
        yr = Yr[:q, :].T  # (T-p+1 x q)
        for i in range(1, p):
            yr = np.vstack([Yr[i*q:(i+1)*q, 0:1].T, yr])
        yr = yr - yr.mean(axis=0)  # demean

        # Re-estimasi VAR
        Ar, SIGMAr, Ur_new, Vr = olsvarc(yr, p)

        # Bias correction jika stasioner
        if not np.any(np.abs(np.linalg.eigvals(Ar)) >= 1):
            Ar = asybc(Ar, SIGMAr, T, p)

        # Selalu simpan IRF (sesuai MATLAB asli)
        IRFr = irfvar(Ar, SIGMAr[:q, :q], p, h, q)
        IRFrmat[j, :] = IRFr.T.flatten(order='C')

    print(f"Bootstrap selesai: {nrep} replikasi")

    CI = np.percentile(IRFrmat, [5, 95], axis=0)
    return CI


def point_test(data_path, p=2, h=12, nrep=1000, seed=42):
    """Main: load data, estimasi, bootstrap, export CSV."""
    y       = pd.read_csv(data_path, header=None).values  # (T x q)
    T, q    = y.shape
    y_dm    = y - y.mean(axis=0)  # demean

    # Estimasi VAR
    A, SIGMA, U, V = olsvarc(y_dm, p)

    # Bias correction
    if not np.any(np.abs(np.linalg.eigvals(A)) >= 1):
        A = asybc(A, SIGMA, T, p)

    # Point estimate IRF
    IRF     = irfvar(A, SIGMA[:q, :q], p, h, q)
    IRF_mat = IRF.T  # (h+1 x q^2)

    # Bootstrap CI
    CI       = boot(A, U, y_dm, V, p, h, q, nrep=nrep, seed=seed)
    CI_lower = CI[0].reshape(h+1, q**2)
    CI_upper = CI[1].reshape(h+1, q**2)

    # Export
    pd.DataFrame(IRF_mat).to_csv('irf_point_python.csv', index=False, header=False)
    pd.DataFrame(CI_lower).to_csv('ci_lower_python.csv', index=False, header=False)
    pd.DataFrame(CI_upper).to_csv('ci_upper_python.csv', index=False, header=False)

    print("Saved: irf_point_python.csv, ci_lower_python.csv, ci_upper_python.csv")
    print("\nIRF point estimate (first 5 horizons):")
    print(np.round(IRF_mat[:5], 4))
    print("\nCI lower (first 5 horizons):")
    print(np.round(CI_lower[:5], 4))
    print("\nCI upper (first 5 horizons):")
    print(np.round(CI_upper[:5], 4))

    return IRF_mat, CI_lower, CI_upper


IRF_mat, CI_lower, CI_upper = point_test('test_data_var.csv', p=2, h=12)