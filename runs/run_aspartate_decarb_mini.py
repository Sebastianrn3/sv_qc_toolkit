#l-aspartate alpha-decarboxylase
from dataclasses import replace
import settings.jobs.asp_1UHE as job
from runs.run_cocaine2 import xyz_start
from utils.brute_multiscan.pull_builder import build_image_groups
from utils.brute_multiscan.relax_images import run_and_record_interpolated_images_relaxation
from utils.brute_multiscan.score_bruteforce import brute_force_paths, analyze_chain_triplets
from utils.helpers.kabsch import evaluate_fixed_atoms_stability
from utils.io.npz_io import load_all_npz_dict
from utils.io.xyz_io import save_chain_xyz
from utils.statistics.clean_pool_outliers import regenerate_raw_npz_to_clean
from utils.statistics.plotgen import plot_final_energies, plot_relaxation_energies, \
    plot_all_relaxation_step_sizes_in_one_figure
import numpy as np
# =======JOB SETTINGS
interpolation_method = "linear" #"linear" or "idpp"
image_pool_selection = "rmds" #"rmsd" or "indices"
n_interpolated = 7
k_select = 7

xyz_start = job.R_XYZ
xyz_end = job.P_XYZ
job.CFG = replace(job.CFG, jobname=f"asp_decarb_n{n_interpolated}_{interpolation_method}_04")

print(f"job {job.CFG}")
print(f"interpolation method: {interpolation_method}")
print(f"k selection method: {image_pool_selection}")
# =======BRUTE-FORCE-FUNCTIONS
def read_final_energies_from_npz(folder=job.CFG.opt_folder / job.CFG.jobname):
    result_dict = load_all_npz_dict(folder)

    final_energies = [result_dict["relaxation_0R.npz"]["energies"][-1]]
    for i in range(1, len(result_dict)-1):
        final_energies += [result_dict[f"relaxation_{i}of{len(result_dict)-2}.npz"]["energies"][-1]]
    final_energies += [result_dict[f"relaxation_{len(result_dict)-1}P.npz"]["energies"][-1]]

    return [float(x) for x in final_energies]
def raw_npz_folder(cfg):
    return cfg.opt_folder / f"{cfg.jobname}_raw"
def clean_npz_folder(cfg):
    return cfg.opt_folder / cfg.jobname
def plot_tag(raw=False):
    return "_raw" if raw else ""
def make_relaxation_plots(cfg, raw=False):
    folder = raw_npz_folder(cfg) if raw else clean_npz_folder(cfg)
    tag = plot_tag(raw)
    npz = load_all_npz_dict(folder)
    energies = read_final_energies_from_npz(folder)

    plot_final_energies(
        final_energies=energies,
        cfg=cfg,
        save_folder=cfg.analysis_folder,
        title=f"final_energies_{cfg.jobname}{tag}",
        filename=f"final_energies_{cfg.jobname}{tag}.png",
        show=False,
    )

    plot_relaxation_energies(
        result_dict=npz,
        save_folder=cfg.analysis_folder,
        title=f"relax_E_profiles_{cfg.jobname}{tag}",
        filename=f"relax_E_profiles_{cfg.jobname}{tag}.png",
        show=False,
    )

    plot_all_relaxation_step_sizes_in_one_figure(
        result_dict=npz,
        save_folder=cfg.analysis_folder,
        title=f"steps_rmsd_{cfg.jobname}{tag}",
        filename=f"steps_rmsd_{cfg.jobname}{tag}.png",
        metric="rmsd",
    )

    plot_all_relaxation_step_sizes_in_one_figure(
        result_dict=npz,
        save_folder=cfg.analysis_folder,
        title=f"steps_norm_{cfg.jobname}{tag}",
        filename=f"steps_norm_{cfg.jobname}{tag}.png",
        metric="norm",
    )

# ======BRUTE-FORCE PIPELINE
#0 align images by fixed atoms
# res = evaluate_fixed_atoms_stability(xyz_start,xyz_end,job.CFG.fixed_atoms,
#                                      save_aligned_xyz=True,
#                                      folder=job.CFG.inputs_folder,
#                                      filename = "model3_p_aligned")
# print(res)

#1 primary relaxation -> 03_opt/jobname_raw
# run_and_record_interpolated_images_relaxation(
#     reactant_xyz_path=xyz_start,
#     product_xyz_path=xyz_end,
#     interpolation_method=interpolation_method,
#     n_interpolated=n_interpolated,
#     cfg=job.CFG,
#     fixed_atoms=job.CFG.fixed_atoms,
#     npz_subfolder=f"{job.CFG.jobname}_raw",
#     record_interpolation_only = False,
# )

#2 clean outliers -> 03_opt/jobname
# regenerate_raw_npz_to_clean(job.CFG, verbose=True)

#3 plots for both raw and clean datasets
# make_relaxation_plots(job.CFG, raw=True)
# make_relaxation_plots(job.CFG, raw=False)

#4 brute-force (uses data cleaned from outliers)
pool = build_image_groups(
    folder=clean_npz_folder(job.CFG),
    k_select=k_select,
    method="indices",#"rmsd" or "indices"
)


#
_, _, best_full = brute_force_paths(pool)
analyze_chain_triplets(pool, best_full)
#
# npz = load_all_npz_dict(job.CFG.opt_folder / job.CFG.jobname)
# save_chain_xyz(pool,
#                (6, 0, 3, 0, 0, 0, 1, 2, 6),
#                npz['relaxation_0R.npz']['atoms'],
#                folder=job.CFG.geometries_folder,
#                filename= f"best_chain_{job.CFG.jobname}.xyz")
#
#



#============MAIN pipeline
# npz = load_all_npz_dict(clean_npz_folder(job.CFG))
# a = npz["relaxation_0R.npz"]["geoms"][-1].ravel()
# b = npz["relaxation_7of7.npz"]["geoms"][10].ravel()
# print(a@b/(np.linalg.norm(a)*np.linalg.norm(b)))


# a = pool[0][6]["geom"]-pool[2][3]["geom"]
# b = pool[1][0]["grad"]
# a=a.reshape(-1)
# b=b.reshape(-1)
# cos = a@b/(np.linalg.norm(a)*np.linalg.norm(b))
# print(cos)
