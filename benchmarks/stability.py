import numpy
import subprocess
import os
import shutil
import tempfile
import re

import os

def preprocess_raw_pdb(pdb_path: str, occ: float = 1.00, bfac: float = 0.00) -> str:
    """
    Ensure ATOM/HETATM records in a PDB include occupancy and B-factor.
    Pads each line to 80 columns.
    
    Args:
        pdb_path: Path to the input PDB file.
        occ: Occupancy value to insert if missing (default 1.00).
        bfac: B-factor value to insert if missing (default 0.00).
    
    Returns:
        Path to the patched PDB file (with '_patched' suffix).
    """
    pdb_path = os.path.abspath(pdb_path)
    if not os.path.isfile(pdb_path):
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")

    base, ext = os.path.splitext(pdb_path)
    out_path = base + "_patched" + ext

    with open(pdb_path, "r") as fin, open(out_path, "w") as fout:
        for line in fin:
            if line.startswith(("ATOM  ", "HETATM")):
                # Strip newline, pad to avoid index errors
                ln = line.rstrip("\n").ljust(80)
                # Replace occupancy (cols 55–60) and B-factor (cols 61–66)
                ln = ln[:54] + f"{occ:6.2f}{bfac:6.2f}" + ln[66:]
                fout.write(ln[:80] + "\n")
            else:
                fout.write(line)
    
    return out_path

    
def run_foldx_stability(pdb_path):
    # Resolve absolute paths
    pdb_path = os.path.abspath(pdb_path)
    pdb_filename = os.path.basename(pdb_path)
    pdb_basename = os.path.splitext(pdb_filename)[0]

    # Define FoldX paths (adjust if needed)
    foldx_exec = "/home/ubuntu/safegenie2/benchmarks/foldx/foldx_20251231"
    rotabase_path = "/home/ubuntu/safegenie2/benchmarks/foldx/rotabase.txt"

    # Validate input paths
    if not os.path.isfile(pdb_path):
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")
    if not os.path.isfile(foldx_exec):
        raise FileNotFoundError(f"FoldX binary not found: {foldx_exec}")
    if not os.path.isfile(rotabase_path):
        raise FileNotFoundError(f"rotabase.txt not found: {rotabase_path}")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = os.path.dirname(pdb_path)

        # Copy PDB and rotabase.txt to temp dir
        tmp_pdb_path = os.path.join(tmpdir, pdb_filename)
        shutil.copy(pdb_path, tmp_pdb_path)
        shutil.copy(rotabase_path, os.path.join(tmpdir, "rotabase.txt"))

        # Run RepairPDB
        subprocess.run([foldx_exec, "--command=RepairPDB", "--pdb=" + pdb_filename],
                       cwd=tmpdir, check=True)

        repaired_pdb = f"{pdb_basename}_Repair.pdb"

        # Set output tag explicitly for Stability
        output_tag = f"{pdb_basename}_Repair_0"
        subprocess.run([
            foldx_exec,
            "--command=Stability",
            "--pdb=" + repaired_pdb,
            "--output-file=" + output_tag
        ], cwd=tmpdir, check=True)

        # Collect output files
        output_files = []
        for fname in os.listdir(tmpdir):
            if fname.startswith(pdb_basename) or fname.startswith(output_tag):
                src = os.path.join(tmpdir, fname)
                dst = os.path.join(output_dir, fname)
                shutil.move(src, dst)
                output_files.append(dst)

        # Parse Stability fxout
        fxout_path = os.path.join(output_dir, f"{output_tag}_ST.fxout")
        stability_result = parse_stability_fxout(fxout_path)

        print(f"FoldX Stability finished. ΔG folding: {stability_result['dG']:.3f} kcal/mol")
        return {
            "deltaG": stability_result['dG'],
            "breakdown": stability_result['full'],
            "repaired_pdb": os.path.join(output_dir, repaired_pdb),
            "fxout_file": fxout_path,
            "all_outputs": output_files
        }



prot_path = "foldx/pdbs/1bcf_30.pdb"
preprocessed_path = preprocess_raw_pdb(prot_path)
a = run_foldx_stability(preprocessed_path)

print(a)

