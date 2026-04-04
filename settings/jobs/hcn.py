from settings.config import JOBS_DIR, MOPAC_EXE_PATH, DATA_DIR
from settings.models import JobConfig

#folder name under data/jobs/
JOB_ID = "hcn"
JOB_DIR = JOBS_DIR / JOB_ID

#template subfolders
REFERENCES_DIR  = JOB_DIR / "00_references"
INPUT_DIR = JOB_DIR / "01_inputs"
GEOMETRIES_DIR = JOB_DIR / "02_geometries"
OPT_DIR   = JOB_DIR / "03_opt"
NEB_DIR   = JOB_DIR / "04_neb"

#raw inputs
RAW_XYZ_DIR = INPUT_DIR / "xyz_raw"
REACTANT_XYZ = RAW_XYZ_DIR / "HNC.xyz"
PRODUCT_XYZ  = RAW_XYZ_DIR / "NCH.xyz"

#Per-job MOPAC working directory
MOPAC_WORKDIR = OPT_DIR / "mopac"

CFG = JobConfig(
    jobname="HCNv8",
    charge=0,
    unpaired_electrons=0,
    mopac_path=MOPAC_WORKDIR,
    mopac_exe=MOPAC_EXE_PATH,
    folder = DATA_DIR,
)


FIXED_ATOMS_0BASED = [1]
