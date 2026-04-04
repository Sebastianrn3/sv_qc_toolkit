from pathlib import Path
import numpy as np


def build_image_groups(
        folder,
        k_select,
        method="rmsd",
        exclude_ends=True,
        ranges=None,   # list[None | tuple(start, stop)] for each relaxation
):
    folder = Path(folder)
    files = []
    for file_path in folder.glob("relaxation_*.npz"):
        tag, step = parse_relax_tag(file_path)
        files.append((step, tag, file_path))

    files.sort(key=lambda x: x[0])

    groups = []
    pool_indices = []
    gid = 0

    for group_idx, (step, tag, file_path) in enumerate(files):
        with np.load(file_path, mmap_mode="r") as data:
            geoms = data["geoms"]
            energies = data["energies"]
            grads = data["grads"]

            start = 0
            stop = len(geoms) - 1

            if ranges is not None and group_idx < len(ranges) and ranges[group_idx] is not None:
                start, stop = ranges[group_idx]
                start = max(0, int(start))
                stop = min(len(geoms) - 1, int(stop))

            if stop < start:
                idx = np.array([], dtype=int)
            else:
                geoms_local = geoms[start:stop + 1]

                if method == "rmsd":
                    idx_local = select_images_by_rmsd(
                        geoms_local, k_select, exclude_ends=exclude_ends
                    )
                elif method == "indices":
                    idx_local = select_images_by_indices(
                        geoms_local.shape[0], k_select, exclude_ends=exclude_ends
                    )
                else:
                    idx_local = select_images_by_indices(
                        geoms_local.shape[0], k_select, exclude_ends=exclude_ends
                    )

                idx = idx_local + start

            print(f"{file_path.name}: idxs selected: {idx} of {len(geoms)}")

            group = []
            group_pool_indices = []

            for rank, k_in_file in enumerate(idx):
                item = {
                    "gid": gid,
                    "relax_tag": tag,
                    "file_step": step,
                    "k_in_file": int(k_in_file),
                    "select_rank": int(rank),
                    "geom": geoms[k_in_file],
                    "energy": float(energies[k_in_file]),
                    "grad": grads[k_in_file],
                }
                group.append(item)
                group_pool_indices.append(int(k_in_file))
                gid += 1

            groups.append(group)
            pool_indices.append(np.asarray(group_pool_indices, dtype=int))

    return groups, pool_indices


def parse_relax_tag(file_path):
    stem = file_path.stem
    tag = stem.split("_", 1)[1]

    if tag.endswith("R"):
        return tag, int(tag[:-1])

    if "of" in tag:
        x, _ = tag.split("of", 1)
        return tag, int(x)

    if tag.endswith("P"):
        return tag, int(tag[:-1])

    raise ValueError(f"Cannot parse relax tag from {file_path}")


def select_images_by_indices(n_images, k_select, exclude_ends=False):
    if k_select <= 0 or n_images <= 0:
        return np.array([], dtype=int)

    if exclude_ends:
        full_idx = select_images_by_indices(n_images, k_select + 2, exclude_ends=False)
        if len(full_idx) <= 2:
            return np.array([], dtype=int)
        return full_idx[1:-1]

    if n_images <= k_select:
        return np.arange(n_images, dtype=int)

    idx = np.linspace(0, n_images - 1, k_select)
    idx = idx.round().astype(int)
    idx = np.unique(idx)

    idx[0] = 0
    idx[-1] = n_images - 1
    idx = np.unique(idx)

    if len(idx) < k_select:
        missing = np.setdiff1d(np.arange(n_images), idx)
        need = k_select - len(idx)
        idx = np.sort(np.concatenate([idx, missing[:need]]))

    return idx.astype(int)


def select_images_by_rmsd(images, k_select, exclude_ends=False):
    def rmsd(geom1, geom2):
        diff = geom1 - geom2
        return np.sqrt(np.mean(np.sum(diff * diff, axis=1)))

    n_images = len(images)

    if k_select <= 0 or n_images <= 0:
        return np.array([], dtype=int)

    if exclude_ends:
        full_idx = select_images_by_rmsd(images, k_select + 2, exclude_ends=False)
        if len(full_idx) <= 2:
            return np.array([], dtype=int)
        return full_idx[1:-1]

    if n_images <= k_select:
        return np.arange(n_images, dtype=int)

    step_distance = np.zeros(n_images - 1, dtype=float)
    for i in range(n_images - 1):
        step_distance[i] = rmsd(images[i], images[i + 1])

    accumulated_distance = np.zeros(n_images, dtype=float)
    accumulated_distance[1:] = np.cumsum(step_distance)
    total = accumulated_distance[-1]

    if total < 1e-10:
        idx = np.linspace(0, n_images - 1, k_select)
        idx = np.round(idx).astype(int)
        idx = np.unique(idx)

        if len(idx) > 0:
            idx[0] = 0
            idx[-1] = n_images - 1
            idx = np.unique(idx)

        if len(idx) < k_select:
            missing = np.setdiff1d(np.arange(n_images), idx)
            need = k_select - len(idx)
            idx = np.sort(np.concatenate([idx, missing[:need]]))

        return idx.astype(int)

    targets = np.linspace(0.0, total, k_select)
    idx = []

    for t in targets:
        j = np.argmin(np.abs(accumulated_distance - t))
        idx.append(j)

    idx = np.array(idx, dtype=int)
    idx = np.unique(idx)

    if len(idx) > 0:
        idx[0] = 0
        idx[-1] = n_images - 1
        idx = np.unique(idx)

    if len(idx) < k_select:
        all_idx = np.arange(n_images, dtype=int)
        missing = np.setdiff1d(all_idx, idx)
        need = k_select - len(idx)

        selected_s = accumulated_distance[idx]
        candidates = []

        for j in missing:
            dist = np.min(np.abs(accumulated_distance[j] - selected_s))
            candidates.append((dist, j))

        candidates.sort(reverse=True)
        extra = [j for dist, j in candidates[:need]]
        idx = np.sort(np.concatenate([idx, extra]))

    return idx.astype(int)

from pathlib import Path
import numpy as np

def build_secondary_ranges_from_best(
    best_full,
    pool_indices,
    folder,
    min_window_size,
    half_window_expansion=0,
):
    """
    best_full[g]      = selected rank inside group g
    pool_indices[g]   = original image indices selected for group g
    folder            = folder with relaxation_*.npz
    min_window_size   = at least this many images in each zoom window
    """

    folder = Path(folder)

    files = []
    for file_path in folder.glob("relaxation_*.npz"):
        tag, step = parse_relax_tag(file_path)
        files.append((step, tag, file_path))
    files.sort(key=lambda x: x[0])

    n_groups = len(pool_indices)
    ranges = [None] * n_groups

    for g in range(1, n_groups - 1):   # do not zoom endpoints
        chosen_rank = best_full[g]
        chosen_pool = np.asarray(pool_indices[g], dtype=int)

        if len(chosen_pool) == 0:
            ranges[g] = None
            continue

        left_rank = max(chosen_rank - 1, 0)
        right_rank = min(chosen_rank + 1, len(chosen_pool) - 1)

        left = int(chosen_pool[left_rank])
        right = int(chosen_pool[right_rank])
        center = int(chosen_pool[chosen_rank])

        if half_window_expansion > 0:
            dl = center - left
            dr = right - center
            left = center - (dl + half_window_expansion)
            right = center + (dr + half_window_expansion)

        # total number of images in this relaxation
        _, _, file_path = files[g]
        with np.load(file_path, mmap_mode="r") as data:
            n_total = len(data["geoms"])

        left = max(0, left)
        right = min(n_total - 1, right)

        # enlarge window until it has at least min_window_size images
        current_size = right - left + 1
        need = min_window_size - current_size

        if need > 0:
            add_left = need // 2
            add_right = need - add_left

            left -= add_left
            right += add_right

            # clamp
            if left < 0:
                right += -left
                left = 0

            if right > n_total - 1:
                left -= (right - (n_total - 1))
                right = n_total - 1

            left = max(0, left)
            right = min(n_total - 1, right)

        ranges[g] = (int(left), int(right))

    return ranges