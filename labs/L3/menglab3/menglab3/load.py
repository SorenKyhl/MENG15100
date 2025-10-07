import numpy as np
import pandas as pd

def generate_projection_data(
    projection_problem=False, 
    n=250, 
    random_state=42,
    d=2.0,                 # half the center-to-center distance along axis u (total 3D separation = 2d)
    sigma=1.5,             # legacy isotropic std (used when projection_problem=False)
    axis_u=(1.0,1.0,1.0),  # separation axis; will be normalized
    sigma_parallel=0.5,    # std along axis_u  (thin axis of the oblate spheroid)
    sigma_perp=2.0         # std in the plane perpendicular to axis_u (broad, to blur projections)
):
    """
    Generate a 3D dataset with two clusters and return a DataFrame with columns ['x','y','z','cluster'].

    Modes
    -----
    - projection_problem=False (default):
        Classic 'stacked along z': x,y overlap; z separated.

    - projection_problem=True:
        Two oblate spheroids whose symmetry axis is axis_u (default (1,1,1)),
        separated by 2d along axis_u. Variance is small along axis_u (sigma_parallel)
        and larger in the perpendicular plane (sigma_perp), so projections on x/y/z
        *overlap*, while 3D remains well separated.
    """

    def _oblate_covariance(u, sigma_parallel, sigma_perp):
      """
      Build an oblate (pancake) covariance aligned with unit vector u:
        Σ = σ_∥^2 (u uᵀ) + σ_⊥^2 (I - u uᵀ)
      """
      u = np.asarray(u, dtype=float)
      u = u / np.linalg.norm(u)
      UUT = np.outer(u, u)
      I = np.eye(3)
      return (sigma_parallel**2) * UUT + (sigma_perp**2) * (I - UUT)

    rng = np.random.default_rng(random_state)

    if projection_problem:
        # Normalize the separation axis
        u = np.asarray(axis_u, dtype=float)
        u = u / np.linalg.norm(u)

        # Means positioned symmetrically along u
        mu_A = -d * u
        mu_B =  d * u

        # Oblate covariance aligned with u
        cov = _oblate_covariance(u, sigma_parallel, sigma_perp)

        # Sample
        A = rng.multivariate_normal(mean=mu_A, cov=cov, size=n)
        B = rng.multivariate_normal(mean=mu_B, cov=cov, size=n)

        x1, y1, z1 = A[:, 0], A[:, 1], A[:, 2]
        x2, y2, z2 = B[:, 0], B[:, 1], B[:, 2]

        # Notes:
        # - Center separation is exactly 2d along u (3D distance = 2d).
        # - Along any cardinal axis e_i, mean difference is 2d * (u·e_i).
        #   For u = (1,1,1)/√3, that's 2d/√3 per axis.
        # - Making sigma_perp sufficiently larger than 2d/√3 ensures heavy overlap in x/y/z projections.
        # - Keeping sigma_parallel small makes the clusters tight along the separation axis, preserving 3D separability.

    else:
        # Classic "stacked along z": x,y overlap; z separated
        x1 = rng.normal(0.0, 1.0, n)
        y1 = rng.normal(0.0, 1.0, n)
        z1 = rng.normal(-2.0, 0.6, n)   # lower slab

        x2 = rng.normal(0.0, 1.0, n)
        y2 = rng.normal(0.0, 0.3, n)
        z2 = rng.normal( 2.0, 0.6, n)   # upper slab

    # Combine clusters
    X = np.concatenate([x1, x2])
    Y = np.concatenate([y1, y2])
    Z = np.concatenate([z1, z2])
    labels = np.array(["A"] * n + ["B"] * n)

    # Create DataFrame
    df = pd.DataFrame({
        "x": X,
        "y": Y,
        "z": Z,
        "cluster": labels
    })

    return df

# --- Examples ---
# Oblate clusters separated on (1,1,1), overlapping in x/y/z projections:
# df = generate_projection_data(
#     projection_problem=True,
#     n=1000,
#     d=2.0,
#     axis_u=(1,1,1),
#     sigma_parallel=0.4,   # tight along separation axis
#     sigma_perp=2.2        # broad in perpendicular plane -> cardinal projections overlap
# )

# Classic stacked-z case:
# df = generate_projection_data(projection_problem=False)


# ----------------------------
# Data generator (2D) — clearer separation
# ----------------------------
def generate_PCA_data(
    n_per_cluster=300,
    theta_data_deg=75.0,   # diagonal direction (degrees) of cluster alignment
    sep=8.0,               # ↑ bigger center-to-center distance along the diagonal (was 4.0)
    sigma_parallel=1.0,    # ↓ tighter along diagonal (was 2.0)
    sigma_perp=0.4,        # tight perpendicular spread
    random_state=42
):
    """
    Two anisotropic Gaussian clusters in 2D, aligned with a diagonal direction.
    The combined dataset is mean-zeroed. Returns X, y, u (direction unit vector).
    """
    rng = np.random.default_rng(random_state)

    # Unit vector for the diagonal direction u = (cosθ, sinθ)
    theta = np.radians(theta_data_deg)
    u = np.array([np.cos(theta), np.sin(theta)], dtype=float)
    u /= np.linalg.norm(u)

    # Orthogonal unit vector v (perpendicular to u)
    v = np.array([-u[1], u[0]])

    # Anisotropic covariance aligned with u (‖) and v (⊥)
    I = np.eye(2)
    cov = (sigma_perp**2) * I + (sigma_parallel**2 - sigma_perp**2) * np.outer(u, u)

    # Cluster means are ±(sep/2) * u, symmetric about the origin
    mu_A = -0.5 * sep * u
    mu_B =  0.5 * sep * u

    A = rng.multivariate_normal(mean=mu_A, cov=cov, size=n_per_cluster)
    B = rng.multivariate_normal(mean=mu_B, cov=cov, size=n_per_cluster)

    X = np.vstack([A, B])
    y = np.r_[np.zeros(n_per_cluster, dtype=int), np.ones(n_per_cluster, dtype=int)]

    # Mean-zero the combined dataset exactly
    X = X - X.mean(axis=0, keepdims=True)
    return X, y, u

def generate_kmeans_data():
  # generate 1D, three clusters.
  np.random.seed(0) # Set a seed for reproducibility
  data = np.hstack((np.random.normal(-2.0, 0.33, 10), 
                np.random.normal(0.0, 0.33, 10),
                np.random.normal(2.0, 0.33, 10))) 
  return data
