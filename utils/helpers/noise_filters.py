import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import make_smoothing_spline


def _arc_parameter(geoms):#i-[0, 1]
    n_steps = geoms.shape[0]

    flat = geoms.reshape(n_steps, -1)
    step_lengths = np.linalg.norm(np.diff(flat, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(step_lengths)])

    if s[-1] == 0.0:
        return np.linspace(0.0, 1.0, n_steps)

    return s / s[-1]


def filter_Savitzky_Golay(geoms, window_length=11, polynomial_order=3): #relax trajectory = [image[atoms[xyz]]], wl odd
    geoms = np.asarray(geoms)
    n_steps, n_atoms, _ = geoms.shape

    flat = geoms.reshape(n_steps, -1)
    flat_smooth = savgol_filter(flat, window_length=window_length, polyorder=polynomial_order, axis=0)

    out = flat_smooth.reshape(n_steps, n_atoms, 3)
    out[0], out[-1] = geoms[0], geoms[-1]

    return out

def filter_smoothing_spline(geoms,lam=None, use_arc_parameter=True):
    geoms = np.asarray(geoms)
    n_steps, n_atoms, _ = geoms.shape

    x = _arc_parameter(geoms) if use_arc_parameter else np.linspace(0, 1, n_steps)

    flat = geoms.reshape(n_steps, -1)
    flat_smooth = np.empty_like(flat)

    for j in range(flat.shape[1]):
        spl = make_smoothing_spline(x, flat[:, j], lam=lam)
        flat_smooth[:, j] = spl(x)

    out = flat_smooth.reshape(n_steps, n_atoms, 3)

    out[0],  out[-1] = geoms[0], geoms[-1]

    return out, x