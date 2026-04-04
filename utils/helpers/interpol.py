import numpy as np
from scipy.optimize import minimize

def interpolate_linearly(xyz_start, xyz_end, n_mid_img):
    images = [xyz_start]
    for k in range(1, n_mid_img+1):
        t = k / (n_mid_img + 1)
        geometry = (1 - t) * xyz_start + t * xyz_end
        images.append(geometry.reshape(-1, 3))
    images.append(xyz_end)
    return np.array(images)

def interpolate_linearly_1d(xyz_start, xyz_end, n_mid_img):
    images = [xyz_start.ravel()]
    for k in range(1, n_mid_img+1):
        t = k / (n_mid_img + 1)
        geometry = (1 - t) * xyz_start + t * xyz_end
        images.append(geometry)
    xyz_end = xyz_end.ravel()
    images.append(xyz_end)
    return np.array(images)


def interpolate_idpp(xyz_start, xyz_end, n_mid_img, fixed_atoms=None):
    n_atoms = xyz_start.shape[0]
    images = [xyz_start]

    active_atoms = [i for i in range(n_atoms) if i not in fixed_atoms]

    linear_images = []
    for k in range(1, n_mid_img + 1):
        t = k / (n_mid_img + 1)
        linear_images.append((1 - t) * xyz_start + t * xyz_end)

    d_start = np.linalg.norm(xyz_start[:, None] - xyz_start[None, :], axis=-1)
    d_end = np.linalg.norm(xyz_end[:, None] - xyz_end[None, :], axis=-1)

    for i, guess_xyz in enumerate(linear_images):
        t = (i + 1) / (n_mid_img + 1)
        d_target = (1 - t) * d_start + t * d_end

        def idpp_objective(active_x_flat):
            x = np.copy(guess_xyz)
            x[active_atoms] = active_x_flat.reshape(-1, 3)
            d_current = np.linalg.norm(x[:, None] - x[None, :], axis=-1)
            error = np.sum((d_current - d_target) ** 2 / (d_target ** 4 + 1e-6))
            return error

        active_guess = guess_xyz[active_atoms].flatten()

        res = minimize(idpp_objective, active_guess, method='L-BFGS-B')

        final_xyz = np.copy(guess_xyz)
        final_xyz[active_atoms] = res.x.reshape(-1, 3)
        images.append(final_xyz)

    images.append(xyz_end)
    return np.array(images)

