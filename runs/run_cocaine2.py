#cocaine esterase deacylation (2 stage)
import settings.jobs.cocaine_stage2 as job
from utils.brute_multiscan.pull_builder import build_image_groups
from utils.brute_multiscan.relax_images import run_and_record_interpolated_images_relaxation
from utils.brute_multiscan.score_bruteforce import brute_force_paths, analyze_chain_triplets
from utils.helpers.interpol import *
from utils.fit_chain.fit_chain_neb_v3 import *
from utils.helpers.kabsch import evaluate_fixed_atoms_stability
from utils.io.npz_io import load_all_npz_dict
from utils.optims.endpoint_optim import *
from dataclasses import replace
from utils.statistics.clean_pool_outliers import regenerate_raw_npz_to_clean
from utils.statistics.plotgen import plot_final_energies, plot_relaxation_energies, plot_all_relaxation_step_sizes_in_one_figure

# =======JOB SETTINGS
interpolation_method = "linear" #"linear" or "idpp"
image_pool_selection = "rmsd" #"rmsd" or "indices"
n_interpolated = 7
k_select = 7

#before align
# xyz_start = job.INT3_XYZ
# xyz_end = job.PROD_XYZ
#xyz_prestart = job.INT2_XYZ

xyz_start = job.INT3_XYZ
xyz_end = job.PROD_XYZ_ALIGNED


job.CFG = replace(job.CFG, jobname=f"cocaine_i3p-n{n_interpolated}_{interpolation_method}_07")

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



# ======BRUTE-FORCE PIPELINE
#0 align images by fixed atoms

align_anchors = [1, 19, 26]
res = evaluate_fixed_atoms_stability(job.INT3_XYZ,
                                     job.PROD_XYZ,
                                     align_anchors,
                                     save_aligned_xyz=True,
                                     folder=job.CFG.inputs_folder,
                                     filename = "PROD_aligned")
print(res)


# 1 primary relaxation -> 03_opt/jobname_raw
run_and_record_interpolated_images_relaxation(
    reactant_xyz_path=xyz_start,
    product_xyz_path=xyz_end,
    interpolation_method=interpolation_method,
    n_interpolated=n_interpolated,
    cfg=job.CFG,
    fixed_atoms=job.CFG.fixed_atoms,
    npz_subfolder=f"{job.CFG.jobname}_raw",
    record_interpolation_only = False,
)

#2 clean outliers -> 03_opt/jobname
regenerate_raw_npz_to_clean(job.CFG, verbose=True)

#3 plots for both raw and clean datasets
# make_relaxation_plots(job.CFG, raw=True)
# make_relaxation_plots(job.CFG, raw=False)

#4 brute-force (uses data without outliers)
pool = build_image_groups(
    folder=clean_npz_folder(job.CFG),
    k_select=k_select,
    method="image_pool_selection",
)



_, _, best_full = brute_force_paths(pool)
analyze_chain_triplets(pool, best_full)
#
#
#
# groups = []
# combo = (0, 0, 0, 0, 0, 0, 0, 0, 0)
# for idx, group in enumerate(pool):
#     groups.append(group[combo[idx]]["geom"])
#
#
# npz = load_all_npz_dict(clean_npz_folder(job.CFG))
# write_xyz_series(
#         atoms=npz['relaxation_0R.npz']['atoms'],
#         list_of_geoms_bohr=groups,
#         file_name="linear_diagnose.xyz",
#         folder=job.CFG.geometries_folder,
#         flatten_all_to_one = True
# )



# save_chain_xyz(pool, (6, 1, 0, 0, 1, 0, 0, 1, 6), npz['relaxation_0R.npz']['atoms'], "amy.xyz")
# npz = load_all_npz_dict(clean_npz_folder(job.CFG))
# a = npz["relaxation_0R.npz"]["geoms"][-1].ravel()
# b = npz["relaxation_7of7.npz"]["geoms"][10].ravel()
# print(a@b/(np.linalg.norm(a)*np.linalg.norm(b)))


# -----PLOTS




# #=======OPTIMIZE_CHAIN
# def run_chain_fit(start_xyz, end_xyz):
#     atoms, geom_int3 = import_xyz(start_xyz)
#     _, geom_prod = import_xyz(end_xyz)
#
#     geoms = interpolate_linearly(geom_int3, geom_prod, n_mid_img=n_interpolated)
#     write_xyz_series(atoms, geoms, "raw_interpolated_"+job.CFG.jobname, folder=job.GEOMETRIES_DIR,flatten_all_to_one=True)
#
#     images_fitted, energies = fit_chain_neb_ci(geoms, atoms, n_steps=neb_steps, cfg=job.CFG)
#     print(f"Energies of latest images:\n{energies}")
#     write_xyz_series(atoms, images_fitted, file_name=job.CFG.jobname, folder=job.GEOMETRIES_DIR, flatten_all_to_one=True)
#

#============MAIN pipeline
#1 prepare endpoints
# optimize_endpoint(xyz_start, job.CFG.fixed_atoms, job.CFG)
# optimize_endpoint(xyz_end, job.CFG.fixed_atoms, job.CFG) repair - defaultname is optimized_endpoint only

#2 run neb with ci-neb
# run_chain_fit(job.INPUT_DIR / "INT3_opt.xyz", job.CFG.inputs_folder / "PD_opt.xyz")
