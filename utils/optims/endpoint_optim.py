from utils.io.xyz_io import *
from settings.config import SHIFT_LIMIT_ANG
from math import sqrt

from utils.optims.optimize_masked import sci_minimize_multi


def opt_both_endpoints(
        R_path: Path,
        P_path: Path,
        cfg,
        fixed_atoms,
        rigid_groups=None,
        npz_record=False
):
    r_atoms, r_xyz_raw = import_xyz(R_path)
    p_atoms, p_xyz_raw = import_xyz(P_path)
    assert np.array_equal(r_atoms, p_atoms)

    r_xyz_optimized, res_r = sci_minimize_multi(
        r_atoms,
        r_xyz_raw,
        cfg,
        fixed_atoms,
        rigid_groups,
        npz_record,
        nr="R_optimized"
    )

    p_xyz_optimized, res_p = sci_minimize_multi(
        p_atoms,
        p_xyz_raw,
        cfg,
        fixed_atoms,
        rigid_groups,
        npz_record,
        nr="P_optimized"
    )

    write_xyz(r_atoms, r_xyz_optimized, file_name="R_opt", path=cfg.inputs_folder)
    write_xyz(p_atoms, p_xyz_optimized, file_name="P_opt", path=cfg.inputs_folder)

    if check_fixed_atom_shifts(r_xyz_optimized, p_xyz_optimized, fixed_atoms):
        print(f"Too large >{SHIFT_LIMIT_ANG} shifts found among reactant and product fixed atoms. Try Kabsch...")
        assert False


def optimize_endpoint(
        xyz_path: Path,
        fixed_atoms,
        cfg,
        rigid_groups=None,
        npz_record=False
):
    atoms, geometry_raw = import_xyz(xyz_path)

    geom_new, res = sci_minimize_multi(
        atoms,
        geometry_raw,
        cfg,
        fixed_atoms,
        rigid_groups,
        npz_record,
        nr="optimized",
    )
    write_xyz(atoms, geom_new, file_name="optimized_endpoint", path=cfg.inputs_folder)
    print("Endpoint optimized")

def check_fixed_atom_shifts(r_xyz, p_xyz, fixed_list, shift_limit_ang=SHIFT_LIMIT_ANG):
    r_xyz = np.asarray(r_xyz, dtype=float).reshape(-1, 3)
    p_xyz = np.asarray(p_xyz, dtype=float).reshape(-1, 3)

    limit_bohr = shift_limit_ang / ANGSTROM_PER_BOHR

    fixed0 = [(i-1) for i in fixed_list]
    for atom in fixed0:
        shift = sqrt(
            (r_xyz[atom, 0] - p_xyz[atom, 0]) ** 2 +
            (r_xyz[atom, 1] - p_xyz[atom, 1]) ** 2 +
            (r_xyz[atom, 2] - p_xyz[atom, 2]) ** 2
        )
        if shift > limit_bohr:
            return True
    return False