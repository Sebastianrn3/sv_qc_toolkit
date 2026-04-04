#HCN run protocol
import subprocess
import sys

import settings.jobs.hcn as job
import utils
from utils.helpers.interpol import *
from utils.fit_chain.fit_chain_neb_v3 import *
from utils.optims.endpoint_optim import *
from settings.jobs.hcn import GEOMETRIES_DIR


#========================pipeline
# #=======OPTIMIZE_CHAIN

# def run_chain_fit():
#
#     atoms, geom_0 = import_xyz(GEOMETRIES_DIR / "P_opt.xyz")
#     _, geom_8 =     import_xyz(GEOMETRIES_DIR / "R_opt.xyz")
#
#     geom_3 = geom_0.copy()
#     geom_3[2][1] = geom_3[2][1]+2
#     geom_5 = geom_8.copy()
#     geom_5[2][1] = geom_5[2][1]+2
#     geoms = interpolate_linearly(geom_0, geom_3, n_mid_img=2)
#     seg = interpolate_linearly(geom_3, geom_5, n_mid_img=1)[1:]
#     geoms = np.concatenate([geoms, seg], axis=0)
#     seg = interpolate_linearly(geom_5, geom_8, n_mid_img=2)[1:]
#     geoms = np.concatenate([geoms, seg], axis=0)
#
#     write_xyz_series(atoms, geoms, "raw_interpolated_"+job.CFG.jobname, folder=job.CFG.folder)
#     images_fitted, energies = fit_chain_neb_ci(geoms, atoms, n_steps=5, cfg=job.CFG)
#     write_xyz_series(atoms, images_fitted, file_name=job.CFG.jobname, folder=job.CFG.folder)
#
#     out_dir = Path(job.CFG.folder) / f"{job.CFG.jobname}_{len(images_fitted)}"
#     out_dir.mkdir(parents=True, exist_ok=True)
#
#     png_path = out_dir / "final_bead_energies.png"
#
#     energies_str = " ".join(f"{e:.12f}" for e in energies)
#
#     cmd = [
#         sys.executable,
#         utils.statistics.plot,
#         "-",
#         "--ref", "min",
#         "--unit", "kcal",
#         "--out", str(png_path),
#         "--title", f"{job.CFG.jobname} final bead energies",
#     ]
#     subprocess.run(cmd, input=energies_str, text=True, check=True)
#     print("Saved plot:", png_path)

#1
# opt_both_endpoints(
#     job.REACTANT_XYZ,
#     job.PRODUCT_XYZ,
#     job.CFG,
#     job.FIXED_ATOMS_0BASED,
#     rigid_groups=None
# )
#2
# run_chain_fit()
#3 run

