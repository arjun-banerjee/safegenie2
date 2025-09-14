import sys
import os
from biotite.structure import tm_score, superimpose_structural_homologs
from biotite.structure.io.pdb import PDBFile
import preprocess_pdb
# === checker_2.py (patched parts) ===========================================
import sys
import os
import numpy as np
from biotite.structure import AtomArray, tm_score, superimpose_structural_homologs
from biotite.structure.io.pdb import PDBFile
import preprocess_pdb

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ---------- helpers ----------
def _read_atomarray(pdb_path: str, model: int = 1) -> AtomArray:
    pdb = PDBFile.read(pdb_path)
    return pdb.get_structure(model=model)

def _select_ca(arr: AtomArray) -> AtomArray:
    return arr[arr.atom_name == "CA"]

def _build_ca_correspondence(ref_ca: AtomArray, sam_ca: AtomArray):
    """
    Match CA atoms by (chain_id, res_id, ins_code). Returns matched (ref_sel, sam_sel).
    Ordering is stable so indices correspond 1:1.
    """
    # normalize fields for hashing/sorting
    ref_keys = list(zip(ref_ca.chain_id.astype("U1"),
                        ref_ca.res_id.astype(int),
                        ref_ca.ins_code.astype("U1")))
    sam_keys = list(zip(sam_ca.chain_id.astype("U1"),
                        sam_ca.res_id.astype(int),
                        sam_ca.ins_code.astype("U1")))
    ref_map = {k: i for i, k in enumerate(ref_keys)}
    sam_map = {k: i for i, k in enumerate(sam_keys)}
    common = sorted(set(ref_map).intersection(sam_map), key=lambda x: (x[0], x[1], x[2]))
    if not common:
        return ref_ca[:0], sam_ca[:0]
    ref_idx = [ref_map[k] for k in common]
    sam_idx = [sam_map[k] for k in common]
    return ref_ca[ref_idx], sam_ca[sam_idx]

def _kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """
    Kabsch RMSD between Nx3 sets. Works for N>=2 (rotation underdetermined for N=2,
    but SVD still yields a consistent solution).
    """
    Pc = P - P.mean(axis=0, keepdims=True)
    Qc = Q - Q.mean(axis=0, keepdims=True)
    C = Pc.T @ Qc
    V, S, Wt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(V @ Wt))
    U = V @ np.diag([1, 1, d]) @ Wt
    P_aln = Pc @ U
    diff = P_aln - Qc
    return float(np.sqrt((diff * diff).sum() / P.shape[0]))

# ---------- replacements for your functions ----------
def load_structure(pdb_path):
    """
    Load CA-only AtomArray. (PDBs are already CA-only in your case, but this keeps it explicit.)
    """
    arr = _read_atomarray(pdb_path, model=1)
    return _select_ca(arr)

def compute_tm_score(ref_path, sample_path):
    """
    Try TM on matched CA atoms if we have ≥3 points; otherwise fallback to Kabsch RMSD.
    Returns: positive TM-score in [0,1] or negative RMSD (so the caller can distinguish).
    """
    ref_ca_all = load_structure(ref_path)
    sam_ca_all = load_structure(sample_path)

    ref_ca, sam_ca = _build_ca_correspondence(ref_ca_all, sam_ca_all)
    k = min(ref_ca.array_length(), sam_ca.array_length())
    if k == 0:
        raise ValueError(
            f"No shared CA residues between ref and sample for {os.path.basename(sample_path)}."
        )

    # trim to equal length (they are already 1:1 ordered)
    ref_ca = ref_ca[:k]
    sam_ca = sam_ca[:k]

    if k >= 3:
        try:
            superimposed, _, ref_idx, sam_idx = superimpose_structural_homologs(
                ref_ca, sam_ca, max_iterations=10
            )
            score = tm_score(ref_ca, superimposed, ref_idx, sam_idx)
            print(f"[TM]   {os.path.basename(sample_path)} -> TM={float(score):.4f} (k={k} CA)")
            return float(score)
        except ValueError as e:
            # anchors not found -> fallback below
            pass

    # Fallback path: Kabsch RMSD on CA coords (works for k=2)
    rmsd = _kabsch_rmsd(ref_ca.coord, sam_ca.coord)
    print(f"[RMSD] {os.path.basename(sample_path)} -> RMSD={rmsd:.3f} (k={k} CA)")
    return -rmsd

def compute_all_tm_scores(ref_path, sample_paths):
    """
    Computes scores for each sample:
      - TM-score in [0,1] if TM alignment succeeded,
      - negative RMSD if TM failed or k<3 (returned as -RMSD).
    Prints separate summaries.
    """
    scores = {}
    tm_vals, rmsd_vals = [], []
    skipped = []

    for sample_path in sample_paths:
        print("currently processing:", sample_path)
        try:
            val = compute_tm_score(ref_path, sample_path)
            scores[os.path.basename(sample_path)] = val
            if val >= 0:
                tm_vals.append(val)
            else:
                rmsd_vals.append(-val)
        except Exception as e:
            print(f"[skip] {os.path.basename(sample_path)} -> {e}")
            skipped.append(os.path.basename(sample_path))

    if tm_vals:
        print(f"Average TM-score (n={len(tm_vals)}): {sum(tm_vals)/len(tm_vals):.4f}")
    if rmsd_vals:
        print(f"Average RMSD (fallback, n={len(rmsd_vals)}): {sum(rmsd_vals)/len(rmsd_vals):.3f} Å")
    if not tm_vals and not rmsd_vals:
        print("No comparable structures.")

    if skipped:
        print(f"Skipped {len(skipped)}:", ", ".join(skipped))

    return scores
# ===========================================================================

# The rest of your file (load_pdbs_in_directory, main, etc.) can stay the same.


def load_pdbs_in_directory(directory):
    """
    Load all PDB files in a directory and return a list of their paths.
    """
    pdb_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.pdb')]
    return pdb_files



def main():
    ref_path =  os.path.join(ROOT_DIR, "data/design25/1bcf.pdb")
    # clean and replace
    ref_path_cleaned = preprocess_pdb.preprocess_raw_pdb(ref_path)
    
    # clean entire dir and create sub dir at: pdb_dir/cleaned_pdbs/
    pdb_dir = os.path.join(ROOT_DIR, "results/base/evals_2/motif=1bcf/pdbs")
    pdb_dir_cleaned = preprocess_pdb.preprocess_pdb_dir(pdbs_dir=pdb_dir, out_subdir="cleaned_pdbs", occ=1.00, bfac=0.00) # replaces with defaults
    
    # replace pipeline with cleaned pdbs
    sample_paths = load_pdbs_in_directory(pdb_dir_cleaned)
    scores = compute_all_tm_scores(ref_path_cleaned, sample_paths)
    
    
    # print("TM-scores:", scores)
    print("\n=== Results per file ===")
    for fname, val in scores.items():
        print(f"{fname:30s} score = {val:.6f}")
        # if val >= 0:
        #     print(f"{fname:30s} TM-score = {val:.4f}")
        # else:
        #     print(f"{fname:30s} RMSD     = {-val:.3f} Å (fallback)")

    # ref_path = "/home/ubuntu/safegenie2/data/pdbs/1QLZ.pdb"
    # sample_path = "/home/ubuntu/safegenie2/data/pdbs/7X1U.pdb"
    # score = compute_tm_score(ref_path, sample_path)
    # print(f"TM-score between {os.path.basename(ref_path)} and {os.path.basename(sample_path)}: {score:.4f}")

if __name__ == "__main__":
    main()
