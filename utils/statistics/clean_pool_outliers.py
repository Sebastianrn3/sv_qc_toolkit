import numpy as np
from pathlib import Path
from utils.io.npz_io import  load_all_npz_dict


def filter_global_outliers(energies, sigma=3):
    median_e = np.median(energies)
    std = np.std(energies)
    return energies < (median_e + sigma * std)


def filter_local_outliers(energies, threshold=2.5, window=5):
    mask = np.ones(len(energies), dtype=bool)
    half_w = window // 2

    for i in range(half_w, len(energies) - half_w):
        neighbors = np.delete(energies[i - half_w: i + half_w + 1], half_w)
        local_med = np.median(neighbors)
        local_std = np.std(neighbors) + 1e-8

        if (energies[i] - local_med) / local_std > threshold:
            mask[i] = False
    return mask


def clean_trajectory_with_stats(data, g_sigma=3, l_threshold=2.5, verbose=False):
    geoms = np.asarray(data["geoms"])
    energies = np.asarray(data["energies"])
    grads = np.asarray(data["grads"])

    pre_mask = np.ones(len(energies), dtype=bool)

    first_step_bad = False
    if  energies[2] < energies[1] > energies[0]:
        first_step_bad = True
        pre_mask[1] = False

    geoms = geoms[pre_mask]
    energies = energies[pre_mask]
    grads = grads[pre_mask]

    n_total = len(energies)

    g_mask = filter_global_outliers(energies, sigma=g_sigma)
    n_global = n_total - np.sum(g_mask)

    l_mask = filter_local_outliers(energies, threshold=l_threshold)
    n_local = n_total - np.sum(l_mask)

    final_mask = g_mask & l_mask
    n_final_removed = n_total - np.sum(final_mask)

    cleaned = {
        "atoms": data["atoms"],
        "geoms": geoms[final_mask],
        "energies": energies[final_mask],
        "grads": grads[final_mask],
        "ids": np.arange(np.sum(final_mask), dtype=int),
    }

    stats = {
        "total": n_total,
        "global": int(n_global),
        "local": int(n_local),
        "first_step_bad": int(first_step_bad),
        "final_removed": int(n_final_removed + (1 if first_step_bad else 0)),
    }
    if verbose:
        print(stats)
    return cleaned

def regenerate_npz_without_outliers(src_folder, dst_folder, verbose=True, strict_filtering = False):
    dst_folder.mkdir(parents=True, exist_ok=True)
    raw_npz = load_all_npz_dict(folder=src_folder)
    for filename, data in raw_npz.items():
        clean_data = clean_trajectory_with_stats(data, verbose=verbose)

        np.savez_compressed(
            dst_folder / filename,
            atoms=np.asarray(clean_data["atoms"]),
            geoms=np.asarray(clean_data["geoms"]),
            energies=np.asarray(clean_data["energies"]),
            grads=np.asarray(clean_data["grads"]),
            ids=np.asarray(clean_data["ids"]),
        )

        if verbose:
            print(f"[OK] cleaned: {filename}")

    print(f"All clean npz saved to: {dst_folder}")
    return dst_folder

def regenerate_raw_npz_to_clean(cfg, verbose=True):
    src_folder = Path(cfg.opt_folder) / f"{cfg.jobname}_raw"
    dst_folder = Path(cfg.opt_folder) / cfg.jobname
    return regenerate_npz_without_outliers(src_folder=src_folder, dst_folder=dst_folder, verbose=verbose)