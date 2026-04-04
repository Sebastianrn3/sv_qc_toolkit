from scipy.optimize import  minimize
from utils.optims.rotor import *
from ..io.run_1scf import *
from ..io.npz_io import *

def sci_minimize_multi(
        atoms,
        x0_bohr,
        cfg,
        fixed_atoms=None,
        rigid_groups=None,
        npz_record=False,
        nr="noname",
        npz_subfolder=None,
):
    if type(rigid_groups)==bool: #repair 2
        rigid_groups=None

    fixed_atoms = [] if fixed_atoms is None else fixed_atoms
    rigid_groups = [] if rigid_groups is None else rigid_groups

    fixed_atoms = np.asarray(fixed_atoms)

    n_atoms = len(atoms)


    #sorting rotor's, non-anchor fixed and free atoms:
    mask_rotor = np.zeros(n_atoms, dtype=bool)
    for group in rigid_groups:
        g = np.asarray(group, dtype=int)
        mask_rotor[g] = True

    mask_fixed = np.zeros(n_atoms, dtype=bool)
    if fixed_atoms.size:
        mask_fixed[fixed_atoms] = True

    free_atoms = np.arange(n_atoms, dtype=int)[~mask_rotor & ~mask_fixed]

    rotors_data = prepare_rotors(x0_bohr, rigid_groups, fixed_atoms)

    x0_free_coords = x0_bohr.reshape(-1, 3)[free_atoms].ravel()

    n_rotor_dimensions = sum(g["ndof"] for g in rotors_data)
    x0 = np.concatenate([np.zeros(n_rotor_dimensions), x0_free_coords])

    recorder = NPZImageRecorder(cfg.opt_folder, atoms) if npz_record else None

    def run_1scf(x):
        full_bohr = build_geom_from_rotors(x, x0_bohr, rotors_data, free_atoms).reshape(-1, 3)
        E_Eh, G_Eh_per_bohr = main_mopac(atoms, full_bohr, cfg)
        G_packed = pack_gradients_multi_rotor(
            G_Eh_per_bohr,
            full_bohr,
            rotors_data,
            free_atoms
        )
        if recorder is not None:
            recorder.add(full_bohr, E_Eh, G_Eh_per_bohr)

        return E_Eh, G_packed

    res = minimize(
        fun=run_1scf,
        x0=x0,
        method="L-BFGS-B",
        jac=True,
        options={
            "gtol": 1e-4,#default 1e-5
            "ftol": 1e-8,#defailt 2.2e-9
        }

    )
    full_bohr_opt = build_geom_from_rotors(res.x, x0_bohr,rotors_data, free_atoms)

    print(res)

    if recorder is not None:
        subfolder = npz_subfolder if npz_subfolder is not None else cfg.jobname
        recorder.save_images(subfolder=subfolder, name=f"relaxation_{nr}")
        save_optimize_result_npz(res, folder=cfg.opt_folder, name="opt_result")

    return full_bohr_opt.reshape(-1, 3), res