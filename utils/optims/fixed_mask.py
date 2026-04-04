import numpy as np

def create_free_mask(n_atoms: int, fixed_atoms):
    fixed0 = [(i - 1) for i in fixed_atoms]
    mask = np.ones(n_atoms, dtype=bool)
    for fix in fixed0:
        mask[fix] = False
    return mask