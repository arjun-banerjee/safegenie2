import numpy
import subprocess
import os
import shutil
import tempfile
import re

import os
from pathlib import Path
from typing import Iterable, Tuple

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

def _clean_pdb_lines(lines: Iterable[str], occ: float, bfac: float) -> Iterable[str]:
    for line in lines:
        if line.startswith(("ATOM  ", "HETATM")):
            ln = line.rstrip("\n").ljust(80)                # ensure we have room to slice
            # occupancy (cols 55–60), B-factor (cols 61–66), 1-based PDB spec
            ln = ln[:54] + f"{occ:6.2f}{bfac:6.2f}" + ln[66:]
            yield (ln[:80] + "\n")                          # hard cap to 80 cols + newline
        else:
            yield line

def preprocess_pdb_dir(
    pdbs_dir: str,
    out_subdir: str = "cleaned_pdbs",
    occ: float = 1.00,
    bfac: float = 0.00,
    patterns: Tuple[str, ...] = ("*.pdb", "*.PDB", "*.ent", "*.ENT"),
) -> str:
    """
    Clean all PDB files in `pdbs_dir` and write them into a subfolder in the same dir.

    Output filenames become `name_patched.pdb`.

    Args:
        pdbs_dir: Directory containing input PDB files.
        out_subdir: Name of subfolder to write cleaned PDBs (created if missing).
        occ: Occupancy value for cols 55–60 (default 1.00).
        bfac: B-factor value for cols 61–66 (default 0.00).
        patterns: Filename patterns to match (default: common PDB extensions).

    Returns:
        Absolute path to the output subdirectory.
    """
    in_dir = Path(pdbs_dir).expanduser().resolve()
    if not in_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {in_dir}")

    out_dir = in_dir / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find PDB files
    files = []
    for pat in patterns:
        files.extend(in_dir.glob(pat))

    if not files:
        print(f"[preprocess_pdb_dir] No PDB files found in {in_dir} matching {patterns}.")
        return str(out_dir)

    for src in files:
        stem = src.stem  # filename without extension
        dst = out_dir / f"{stem}_patched.pdb"
        with src.open("r") as fin, dst.open("w") as fout:
            for cleaned in _clean_pdb_lines(fin, occ=occ, bfac=bfac):
                fout.write(cleaned)

    return str(out_dir)

    
# def run_foldx_stability(pdb_path):
#     # Resolve absolute paths
#     pdb_path = os.path.abspath(pdb_path)
#     pdb_filename = os.path.basename(pdb_path)
#     pdb_basename = os.path.splitext(pdb_filename)[0]

#     # Define FoldX paths (adjust if needed)
#     foldx_exec = "/home/ubuntu/safegenie2/benchmarks/foldx/foldx_20251231"
#     rotabase_path = "/home/ubuntu/safegenie2/benchmarks/foldx/rotabase.txt"

#     # Validate input paths
#     if not os.path.isfile(pdb_path):
#         raise FileNotFoundError(f"PDB file not found: {pdb_path}")
#     if not os.path.isfile(foldx_exec):
#         raise FileNotFoundError(f"FoldX binary not found: {foldx_exec}")
#     if not os.path.isfile(rotabase_path):
#         raise FileNotFoundError(f"rotabase.txt not found: {rotabase_path}")

#     with tempfile.TemporaryDirectory() as tmpdir:
#         output_dir = os.path.dirname(pdb_path)

#         # Copy PDB and rotabase.txt to temp dir
#         tmp_pdb_path = os.path.join(tmpdir, pdb_filename)
#         shutil.copy(pdb_path, tmp_pdb_path)
#         shutil.copy(rotabase_path, os.path.join(tmpdir, "rotabase.txt"))

#         # Run RepairPDB
#         subprocess.run([foldx_exec, "--command=RepairPDB", "--pdb=" + pdb_filename],
#                        cwd=tmpdir, check=True)

#         repaired_pdb = f"{pdb_basename}_Repair.pdb"

#         # Set output tag explicitly for Stability
#         output_tag = f"{pdb_basename}_Repair_0"
#         subprocess.run([
#             foldx_exec,
#             "--command=Stability",
#             "--pdb=" + repaired_pdb,
#             "--output-file=" + output_tag
#         ], cwd=tmpdir, check=True)

#         # Collect output files
#         output_files = []
#         for fname in os.listdir(tmpdir):
#             if fname.startswith(pdb_basename) or fname.startswith(output_tag):
#                 src = os.path.join(tmpdir, fname)
#                 dst = os.path.join(output_dir, fname)
#                 shutil.move(src, dst)
#                 output_files.append(dst)

#         # Parse Stability fxout
#         fxout_path = os.path.join(output_dir, f"{output_tag}_ST.fxout")
#         stability_result = parse_stability_fxout(fxout_path)

#         print(f"FoldX Stability finished. ΔG folding: {stability_result['dG']:.3f} kcal/mol")
#         return {
#             "deltaG": stability_result['dG'],
#             "breakdown": stability_result['full'],
#             "repaired_pdb": os.path.join(output_dir, repaired_pdb),
#             "fxout_file": fxout_path,
#             "all_outputs": output_files
#         }



# prot_path = "foldx/pdbs/1bcf_30.pdb"
# preprocessed_path = preprocess_raw_pdb(prot_path)
# a = run_foldx_stability(preprocessed_path)

# print(a)



# ====== PyRosetta fill-in from CA-only ======
def fill_missing_with_pyrosetta_from_ca(
    pdb_path: str,
    out_suffix: str = "_relaxed",
    coord_sigma: float = 1.5,     # Å; softness of CA coordinate restraints
    coord_weight: float = 1.0,    # weight on coordinate_constraint term
    relax_cycles: int = 5,
) -> str:
    """
    Build a full-atom model from a CA-only PDB using PyRosetta:
      1) Parse CA coordinates + residue names (sequence)
      2) Make pose from sequence (fa_standard)
      3) Add coordinate constraints on CA to target positions
      4) Cartesian FastRelax with coord constraints
      5) Dump <input><out_suffix>.pdb and return its path

    NOTE: If your input is already full-atom, prefer normal FastRelax without this builder.
    """
    import math
    import os

    # Lazy import so your module still loads without PyRosetta
    try:
        import pyrosetta
        from pyrosetta import rosetta
    except Exception as e:
        raise ImportError(
            "PyRosetta not installed. Please install a matching PyRosetta wheel for your platform."
        ) from e

    pdb_path = os.path.abspath(pdb_path)
    if not os.path.isfile(pdb_path):
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")

    # --- 1) Parse CA coords + sequence from the CA-only PDB ---
    three_to_one = {
        "ALA":"A","CYS":"C","ASP":"D","GLU":"E","PHE":"F",
        "GLY":"G","HIS":"H","ILE":"I","LYS":"K","LEU":"L",
        "MET":"M","ASN":"N","PRO":"P","GLN":"Q","ARG":"R",
        "SER":"S","THR":"T","VAL":"V","TRP":"W","TYR":"Y",
    }
    ca_xyz = []  # list of (x,y,z)
    seq1 = []
    last_key = None

    with open(pdb_path) as f:
        for ln in f:
            if not ln.startswith("ATOM"):
                continue
            atname = ln[12:16].strip()
            if atname != "CA":
                continue
            resn = ln[17:20].strip().upper()
            chain = ln[21].strip() or "A"
            resi = int(ln[22:26])
            key = (chain, resi)
            # avoid duplicates if multiple altlocs exist
            if last_key == key:
                continue
            last_key = key
            aa = three_to_one.get(resn)
            if aa is None:
                raise ValueError(f"Unsupported residue name '{resn}' at {chain}{resi}; please map it.")
            seq1.append(aa)
            x = float(ln[30:38]); y = float(ln[38:46]); z = float(ln[46:54])
            ca_xyz.append((x,y,z))

    if not seq1 or len(seq1) != len(ca_xyz):
        raise ValueError("Failed to extract CA trace and sequence from PDB.")

    seq = "".join(seq1)

    # --- 2) Init PyRosetta + make pose from sequence (full-atom) ---
    pyrosetta.init(
        "-beta_nov16_cart -relax:cartesian -ex1 -ex2aro "
        "-use_input_sc -ignore_unrecognized_res -no_optH false"
    )
    pose = rosetta.core.pose.Pose()
    rts = rosetta.core.chemical.ChemicalManager.get_instance().residue_type_set("fa_standard")
    rosetta.core.pose.make_pose_from_sequence(pose, seq, rts)

    # --- 3) Add coordinate constraints on CA atoms to target CA positions ---
    # Add a VIRTUAL root so constraints are in a stable absolute frame
    rosetta.core.pose.addVirtualResAsRoot(pose)
    vrt_seqpos = pose.total_residue()  # last residue is VRT
    base = rosetta.core.id.AtomID(1, vrt_seqpos)  # atom 1 of VRT

    HarmonicFunc = rosetta.core.scoring.func.HarmonicFunc
    CoordCst = rosetta.core.scoring.constraints.CoordinateConstraint
    AtomID = rosetta.core.id.AtomID
    Vec = rosetta.numeric.xyzVector_double_t

    for i, (x, y, z) in enumerate(ca_xyz, start=1):
        ca_idx = pose.residue(i).atom_index("CA")
        func = HarmonicFunc(0.0, coord_sigma)  # 0 offset; sd = coord_sigma
        cst = CoordCst(AtomID(ca_idx, i), base, Vec(x, y, z), func)
        pose.add_constraint(cst)

    # --- 4) Cartesian FastRelax with coordinate constraints ---
    sf = rosetta.core.scoring.ScoreFunctionFactory.create_score_function("beta_nov16_cart")
    sf.set_weight(rosetta.core.scoring.coordinate_constraint, coord_weight)

    relax = rosetta.protocols.relax.FastRelax(sf, relax_cycles)
    relax.cartesian(True)
    # Allow bb+chi; default MoveMap of FastRelax is fine for this use
    relax.apply(pose)

    # Remove the virtual root residue before saving (name3 == "VRT")
    if pose.residue_type(pose.total_residue()).name3() == "VRT":
        pose.conformation().delete_residue_slow(pose.total_residue())

    out_path = os.path.splitext(pdb_path)[0] + out_suffix + ".pdb"
    pose.dump_pdb(out_path)
    return out_path


def fill_and_patch_with_pyrosetta(pdb_path: str,
                                  occ: float = 1.00,
                                  bfac: float = 0.00,
                                  **kwargs) -> str:
    """
    Convenience wrapper:
      - Build full-atom model from CA-only input with PyRosetta + relax
      - Patch occupancy/B (1.00 / 0.00) for downstream tools (FoldX)
    kwargs are forwarded to fill_missing_with_pyrosetta_from_ca (e.g., coord_sigma, coord_weight).
    """
    fullatom = fill_missing_with_pyrosetta_from_ca(pdb_path, **kwargs)
    patched = preprocess_raw_pdb(fullatom, occ=occ, bfac=bfac)
    return patched



# 1) From a CA-only PDB → full-atom relaxed PDB + patched columns
patched = fill_and_patch_with_pyrosetta(
    "/home/ubuntu/safegenie2/results/base/outputs/reproduce_2/pdbs/114_3.pdb",
    coord_sigma=1.5,   # tighter (smaller) = follow CA more strictly
    coord_weight=1.0,  # raise to 2–5 if you need closer fit
    relax_cycles=5
)
print("Patched PDB:", patched)

# 2) Then you can run FoldX on `patched`
# a = run_foldx_stability(patched)  # your existing function


