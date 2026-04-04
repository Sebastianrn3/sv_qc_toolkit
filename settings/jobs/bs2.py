from pathlib import Path

JOBNAME=("bc2_ts3-4_v0")

N_IMAGES = 9 #n intermediate images
K_IMAGES = 5 #k selected from relaxation

TS3_XYZ = Path("data\\bc2\\xyz\\TS3.xyz")
TS4_XYZ = Path("data\\bc2\\xyz\\TS4.xyz")

CHARGE = 0
UNPAIRED_ELECTRONS = 10

FIXED_ATOMS_1BASED = [1, 6, 15] #1 - ser, 6 - ser, 15 - his
RIGID_GROUPS_1BASED = []
#----------------
FIXED_ATOMS_0BASED = [i - 1 for i in FIXED_ATOMS_1BASED]
RIGID_GROUPS_0BASED = [[i - 1 for i in g] for g in RIGID_GROUPS_1BASED]
