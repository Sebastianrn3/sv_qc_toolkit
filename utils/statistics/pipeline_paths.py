from pathlib import Path

def raw_npz_folder(cfg) -> Path:
    return Path(cfg.opt_folder) / f"{cfg.jobname}_raw"

def clean_npz_folder(cfg) -> Path:
    return Path(cfg.opt_folder) / cfg.jobname

def plot_tag(raw: bool) -> str:
    return "_raw" if raw else ""