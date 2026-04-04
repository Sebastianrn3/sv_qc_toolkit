import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def plot_final_energies(final_energies, cfg, save_folder=None, title=None, filename=None, show=False):
    e_min = min(final_energies)
    heights = [e - e_min for e in final_energies]

    save_folder = Path(save_folder or cfg.analysis_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    filename = filename or f"final_energies_{cfg.jobname}.png"
    title = title or Path(filename).stem

    fig, ax = plt.subplots(figsize=(8, 5), num=title)

    x = range(len(heights))
    ax.bar(x, heights)

    ax.set_xlabel("№ of Relaxation")
    ax.set_ylabel("Energy above lowest state, Eh")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)

    for i, h in enumerate(heights):
        ax.text(i, h, f"{h:.6f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()

    out_path = save_folder / filename
    fig.savefig(out_path, dpi=200)

    plt.show() if show else plt.close(fig)
    return out_path

def plot_relaxation_energies(result_dict,
    save_folder=None,
    title="Relaxation energies",
    show=False,
    filename=None
):
    def relax_index(name):
        m = re.search(r"relaxation_(\d+)of\d+\.npz", str(name))
        return int(m.group(1)) if m else 10 ** 9

    def safe_name(name):
        return re.sub(r'[\\/*?:"<>|]+', "_", str(name))

    ordered_items = sorted(result_dict.items(), key=lambda kv: relax_index(kv[0]))
    total = len(ordered_items)
    n_plots = total + 1

    fig, axes = plt.subplots(
        n_plots, 1,
        figsize=(10, 3.2 * n_plots),
        num=title
    )

    #combined plot
    ax = axes[0]
    for name, data in ordered_items:
        energies = np.asarray(data["energies"], dtype=float)
        label = Path(name).stem
        ax.plot(np.arange(len(energies)), energies, marker="o", ms=3, label=label)

    ax.set_title("All relaxations")
    ax.set_xlabel("Relaxation step")
    ax.set_ylabel("Energy, Eh")
    ax.grid(True, alpha=0.3)
    if total <= 15:
        ax.legend(fontsize=8)

    #separate trajectories
    for i, (name, data) in enumerate(ordered_items):
        energies = np.asarray(data["energies"], dtype=float)
        ax = axes[i + 1]

        if i == 0:
            label = "Reactant"
        elif i == total - 1:
            label = "Product"
        else:
            label = f"Interpolant {i}"

        ax.plot(np.arange(len(energies)), energies, marker="o", ms=4)
        ax.set_title(label)
        ax.set_xlabel("Relaxation step")
        ax.set_ylabel("Energy (Eh)")
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    out_path = None
    if save_folder is not None:
        save_folder = Path(save_folder)
        save_folder.mkdir(parents=True, exist_ok=True)

        if filename is None:
            filename = safe_name(title) + ".png"

        out_path = save_folder / filename
        fig.savefig(out_path, dpi=200)

    plt.show() if show else plt.close(fig)

    return out_path

def plot_all_relaxation_step_sizes_in_one_figure(
    result_dict,
    save_folder=None,
    title="Relaxation step sizes",
    filename=None,
    width=14,
    height_per_plot=4.8,
    metric="rmsd",   #"rmsd"/"norm"
):

    def relax_index(name):
        m = re.search(r"relaxation_(\d+)of\d+\.npz", str(name))
        return int(m.group(1)) if m else 10**9

    def safe_name(name):
        return re.sub(r'[\\/*?:"<>|]+', "_", str(name))

    def as_xyz(geom):
        arr = np.asarray(geom, dtype=float)
        if arr.ndim == 2 and arr.shape[1] == 3:
            return arr
        if arr.ndim == 1 and arr.size % 3 == 0:
            return arr.reshape(-1, 3)

    def step_sizes_from_geoms(geoms, metric_name="rmsd"):
        vals = []
        for i in range(1, len(geoms)):
            g_prev = as_xyz(geoms[i - 1])
            g_curr = as_xyz(geoms[i])

            diff = g_curr - g_prev

            if metric_name == "rmsd":
                val = float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))
            if metric_name == "norm":
                val = float(np.linalg.norm(diff.reshape(-1)))

            vals.append(val)
        return np.array(vals, dtype=float)

    ordered_items = sorted(result_dict.items(), key=lambda kv: relax_index(kv[0]))
    total = len(ordered_items)

    n_plots = total + 1
    fig_height = height_per_plot * n_plots

    fig, axes = plt.subplots(
        n_plots, 1,
        figsize=(width, fig_height),
        num=title
    )
    if n_plots == 1:
        axes = [axes]

    ylabel = "Step RMSD" if metric == "rmsd" else "||Δx||"

    ax = axes[0]
    for name, data in ordered_items:
        geoms = np.asarray(data["geoms"], dtype=float)
        steps = step_sizes_from_geoms(geoms, metric_name=metric)
        label = Path(name).stem
        ax.plot(np.arange(1, len(steps) + 1), steps, marker="o", ms=4, label=label)

    ax.set_title("All relaxations", fontsize=14)
    ax.set_xlabel("Relaxation step", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, alpha=0.3)
    if total <= 15:
        ax.legend(fontsize=9)

    for i, (name, data) in enumerate(ordered_items):
        geoms = np.asarray(data["geoms"], dtype=float)
        steps = step_sizes_from_geoms(geoms, metric_name=metric)
        ax = axes[i + 1]

        if i == 0:
            label = "R"
        elif i == total - 1:
            label = "P"
        else:
            label = f"Interpolant {i}"

        ax.plot(np.arange(1, len(steps) + 1), steps, marker="o", ms=5)
        ax.set_title(label, fontsize=14)
        ax.set_xlabel("Relaxation step", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=18)
    fig.subplots_adjust(hspace=0.55, top=0.97, bottom=0.04)

    out_path = None
    if save_folder is not None:
        save_folder = Path(save_folder)
        save_folder.mkdir(parents=True, exist_ok=True)

        if filename is None:
            filename = safe_name(title) + ".png"

        out_path = save_folder / filename
        fig.savefig(out_path, dpi=220, bbox_inches="tight")

    return out_path
