import numpy as np
from scipy.optimize import linear_sum_assignment

def onehot(k, K):
    return np.arange(K) == k


def convex_combo(x0, x1, step_size):
    return (1 - step_size) * x0 + step_size * x1


def compute_state_overlap(z1, z2):
    assert z1.dtype == int and z2.dtype == int
    assert z1.shape == z2.shape

    z1_vals = np.unique(z1)
    z2_vals = np.unique(z2)
    K1 = len(z1_vals)
    K2 = len(z2_vals)

    overlap = np.zeros((K1, K2))
    for k1, z1_val in enumerate(z1_vals):
        for k2, z2_val in enumerate(z2_vals):
            overlap[k1, k2] = np.sum((z1 == z1_val) & (z2 == z2_val))
    return overlap, z1_vals, z2_vals


def permute_to_match(z1, z2):
    """
    Return a permuted copy of z2 to best match z1
    """
    overlap, z1_vals, z2_vals = compute_state_overlap(z1, z2)
    K1, K2 = overlap.shape
    perm_z1, perm_z2 = linear_sum_assignment(-overlap)

    # Pad permutation if K1 < K2
    if K1 < K2:
        # Target sequence (z1) has fewer elements than given sequence (z2)
        # Line up the ones that match best and give the remainder labels
        # from K1, ..., K2-1
        unused = np.array(list(set(np.arange(K2)) - set(perm_z2)))
        perm_z2 = np.concatenate((perm_z2, unused))

    new_z2 = -1000 * np.ones_like(z2)
    for z1_ind, z2_ind in zip(perm_z1, perm_z2):
        z1_val = z1_vals[z1_ind]
        z2_val = z2_vals[z2_ind]
        new_z2[z2 == z2_val] = z1_val

    if K1 < K2:
        for k, z2_ind in enumerate(perm_z2[K1:K2]):
            z2_val = z2_vals[z2_ind]
            new_z2[z2 == z2_val] = K1 + k

    return new_z2
