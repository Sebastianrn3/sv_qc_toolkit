#l-aspartate alpha-decarboxylase
from dataclasses import replace

from numpy import ndim

import settings.jobs.asp_1UHE as job
from utils.brute_multiscan.pull_builder import build_image_groups, build_secondary_ranges_from_best
from utils.brute_multiscan.relax_images import run_and_record_interpolated_images_relaxation
from utils.brute_multiscan.score_bruteforce import brute_force_paths, analyze_chain_triplets
from utils.helpers.kabsch import evaluate_fixed_atoms_stability
from utils.helpers.noise_filters import filter_Savitzky_Golay, filter_smoothing_spline
from utils.io.combo_io import form_combo_report, form_relaxations_report, print_best_chain_image_positions
from utils.io.npz_io import load_all_npz_dict
from utils.io.xyz_io import  write_xyz, write_xyz_series
from utils.statistics.clean_pool_outliers import regenerate_raw_npz_to_clean
from utils.statistics.plotgen import plot_final_energies, plot_relaxation_energies, \
    plot_all_relaxation_step_sizes_in_one_figure
import numpy as np

# =======JOB SETTINGS
interpolation_method = "idpp" #"linear" or "idpp"
image_pool_selection = "rmsd" #"rmsd" or "indices"
n_interpolated = 7

exclude_ends = 1 #0 - include ends, 1 - exclude latest end, 2 - exclude both start and end

brute_force_laps = 3 #1 - no zoom, 2 - single zoom, 3 - double zoom eg 2 consecutive zooms
k_select = 4
k_select_2nd = k_select
k_select_3rd = k_select_2nd



xyz_start = job.R_XYZ
xyz_end = job.P_XYZ
job.CFG = replace(job.CFG, jobname=f"asp_decarb_n{n_interpolated}_{interpolation_method}_08")

print(f"job {job.CFG.jobname}")
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

# form_relaxations_report(
#     folder=job.CFG.opt_folder / job.CFG.jobname,
#     out_path=job.CFG.analysis_folder / f"relaxations_{job.CFG.jobname}.txt",
# )



# ======BRUTE-FORCE PIPELINE
#0 align images by fixed atoms
# res = evaluate_fixed_atoms_stability(xyz_start,xyz_end,job.CFG.fixed_atoms,
#                                      save_aligned_xyz=True,
#                                      folder=job.CFG.inputs_folder,
#                                      filename = "model3_p_aligned")
# print(res)

#1 primary relaxation -> 03_opt/jobname_raw
# interpolation_zero_set = run_and_record_interpolated_images_relaxation(
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
# regenerate_raw_npz_to_clean(job.CFG, verbose=True, strict=True)

#3 plots for both raw and clean datasets
# make_relaxation_plots(job.CFG, raw=True)
# make_relaxation_plots(job.CFG, raw=False)



# 4+ form pool -> brute-force laps with optional zooming
k_select_laps = [k_select, k_select_2nd, k_select_3rd]

def get_k_for_lap(lap_index):
    if lap_index < len(k_select_laps):
        return k_select_laps[lap_index]
    return k_select_laps[-1]

pool = None
pool_indices = None
best_full = None
best_result = None

for lap in range(brute_force_laps):
    lap_no = lap + 1
    k_this = get_k_for_lap(lap)

    print(f"\nLAP {lap_no} " + "=" * 60)

    if lap == 0:
        pool, pool_indices = build_image_groups(
            folder=clean_npz_folder(job.CFG),
            k_select=k_this,
            method=image_pool_selection,
        )
        print("primary pool created")
    else:
        zoom_ranges = build_secondary_ranges_from_best(
            best_full=best_full,
            pool_indices=pool_indices,
            folder=clean_npz_folder(job.CFG),
            min_window_size=k_this,
            half_window_expansion=0,
        )
        print("zoom ranges:", zoom_ranges)

        pool, pool_indices = build_image_groups(
            folder=clean_npz_folder(job.CFG),
            k_select=k_this,
            method=image_pool_selection,
            ranges=zoom_ranges,
            exclude_ends=False,
        )
        print(f"zoom pool created for lap {lap_no}")

    best_result = brute_force_paths(pool)
    best_score, _, best_full = best_result

    print(f"Best score after lap {lap_no}: {best_score:.6f}")
    print(f"Best full combo after lap {lap_no}: {best_full}")

# 5 print brute-force summary

npz = load_all_npz_dict(clean_npz_folder(job.CFG))
atoms_list = npz["relaxation_0R.npz"]["atoms"]

print_best_chain_image_positions(pool, best_full, npz, inner_only=True)

form_combo_report(pool, best_full, job.CFG, atoms_list)






# geom1 = pool[3][3]["geom"]
# geom2 = pool[1][3]["geom"]
# grad23 =pool[2][4]["grad"]
#
# distance = np.linalg.norm(geom1-geom2)

# print("answer is ",(geom2-geom1).ravel()@grad23.ravel()/(np.linalg.norm((geom2-geom1).ravel())*np.linalg.norm(grad23.ravel())))

# print("distance", distance)
# print("angle", angle)

#
# for i in range(len(atoms_list)):
#     if np.linalg.norm(geom2[i]-geom1[i])> 1.e-4 and np.linalg.norm(grad23[i])> 1.e-4:
#         print(i, (geom2[i]-geom1[i])@grad23[i]/(np.linalg.norm(geom2[i]-geom1[i])*np.linalg.norm(grad23[i])))


# geoms_list = npz["relaxation_4of7.npz"]["geoms"]
#
# window_length=21
# polynomial_order=3
#
# def smooth_and_record():
#     #unfiltered record
#     # write_xyz_series(atoms_list, geoms_list, f"unfiltered_{job.CFG.jobname}", folder=job.CFG.geometries_folder)
#
#     #filter Savitsky-Golay
#     geoms_sg = filter_Savitzky_Golay(geoms_list, window_length=window_length, polynomial_order=polynomial_order)
#     write_xyz_series(atoms_list, geoms_sg, f"filtered_SG{window_length}p{polynomial_order}_{job.CFG.jobname}", folder=job.CFG.geometries_folder)
#
#     #filter smoothing spline
#     # geoms_ss, _ = filter_smoothing_spline(geoms_list, lam=None)
#     # write_xyz_series(atoms_list, geoms_ss, f"filtered_spline_{job.CFG.jobname}", folder=job.CFG.geometries_folder)
# smooth_and_record()










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
