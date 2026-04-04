from utils.helpers.interpol import interpolate_linearly, interpolate_idpp
from utils.optims.optimize_masked import *
from utils.io.xyz_io import *
import sys
from contextlib import redirect_stdout, redirect_stderr

class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()

def run_and_record_interpolated_images_relaxation(
        reactant_xyz_path,
        product_xyz_path,
        interpolation_method, #linear or idpp
        n_interpolated,
        cfg,
        fixed_atoms,
        rigid_groups=None,
        npz_subfolder=None,
        record_interpolation_only = False,
):
    r_atoms, r_geom = import_xyz(reactant_xyz_path)
    p_atoms, p_geom = import_xyz(product_xyz_path)

    assert np.array_equal(r_atoms, p_atoms)

    if interpolation_method=="linear":
        geom_set = interpolate_linearly(r_geom, p_geom, n_interpolated)
    elif interpolation_method=="idpp":
        geom_set = interpolate_idpp(r_geom, p_geom, n_interpolated, fixed_atoms)
    else:
        raise ValueError(f"Invalid interpolation method chosen: {interpolation_method}")

    if record_interpolation_only:
        return geom_set
    write_xyz_series(r_atoms,
                     geom_set,
                     file_name=f"{cfg.jobname}_flatten",
                     folder= cfg.geometries_folder,
                     flatten_all_to_one=True
                     )

    if record_interpolation_only:
        print("passed")
        return

    target_npz_subfolder = npz_subfolder or f"{cfg.jobname}_raw"

    n_of_relaxation=0
    total=len(geom_set)
    res_energies_set = []

    run_folder = cfg.geometries_folder / cfg.jobname
    run_folder.mkdir(parents=True, exist_ok=True)

    log_path = run_folder / "log.txt"

    with open(log_path, "a", encoding="utf-8") as log_file:
        tee_out = Tee(sys.stdout, log_file)
        tee_err = Tee(sys.stderr, log_file)

        with redirect_stdout(tee_out), redirect_stderr(tee_err):
            if record_interpolation_only:
                print("passed")
                return

            target_npz_subfolder = npz_subfolder or f"{cfg.jobname}_raw"

            n_of_relaxation = 0
            total = len(geom_set)
            res_energies_set = []

            for relaxation in geom_set:
                print(n_of_relaxation)

                if n_of_relaxation == 0:
                    nr = "0R"
                elif n_of_relaxation == (total - 1):
                    nr = f"{total-1}P"
                else:
                    nr = f"{n_of_relaxation}of{total-2}"

                minimized_geom, res = sci_minimize_multi(
                    atoms=r_atoms,
                    x0_bohr=relaxation,
                    cfg=cfg,
                    fixed_atoms=fixed_atoms,
                    rigid_groups=rigid_groups,
                    npz_record=True,
                    nr=nr,
                    npz_subfolder=target_npz_subfolder,
                )
                res_energies_set.append(res.fun)

                write_xyz(
                    r_atoms,
                    minimized_geom,
                    cfg.jobname + "_" + str(n_of_relaxation),
                    run_folder
                )

                n_of_relaxation += 1

            print("all relaxations done", res_energies_set)