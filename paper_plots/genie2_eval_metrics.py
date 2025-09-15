#!/usr/bin/env python3
"""
Genie 2-style evaluation metrics for a folder of generated PDBs.

Implements the paper's Section 4.1 metrics:
- Designability (ProteinMPNN -> 8 sequences -> ESMFold -> scRMSD <= 2Å AND mean pLDDT >= 70)
- Diversity (single-linkage clustering on *designable* structures with TM-score >= 0.6; report #clusters / N)
- F1 score (harmonic mean of pstructures and pclusters)
- Novelty (vs. reference sets, e.g., PDB and AFDB): cluster *novel* designs (TM <= 0.5 to any ref),
  then report #novel_clusters / N

Notes
-----
1) Inverse folding (ProteinMPNN):
   - This script supports two sequence designers:
       a) ProteinMPNN (preferred; paper uses this)
       b) ESM-IF1 (fallback if ProteinMPNN not available)
   - Select via --seq_designer [proteinmpnn|esm_if]. Default tries ProteinMPNN, falls back to ESM-IF if import/path fails.
   - For ProteinMPNN, provide --proteinmpnn_repo path (root of the repo) or ensure it is importable.

2) Structure prediction (ESMFold):
   - Uses the Facebook AI "esm" package (pip install esm) and runs locally.
   - GPU recommended (set --device cuda:0), CPU works but is slow.

3) TM-score and clustering:
   - We prefer the "TMalign" binary if present in PATH or provided via --tmalign.
   - Otherwise we fall back to biotite.structure.tm_score + Kabsch alignment for an approximate TM-score.
   - Clustering is single-linkage by building a graph where edges connect designs with TM-score >= 0.6,
     then taking connected components.

4) Caching:
   - Sequences, ESMFold PDBs, and per-sample JSON are cached under --workdir to avoid recomputation.

5) Outputs:
   - summary.json : overall metrics (pstructures, pclusters, F1, novelty_pdb, novelty_afdb)
   - per_sample.csv : per-sample measurements and decisions

Usage
-----
python genie2_eval_metrics.py \
  --input_dir path/to/generated_pdbs \
  --workdir runs/checkpoint1_eval \
  --seq_designer proteinmpnn \
  --proteinmpnn_repo /path/to/ProteinMPNN \
  --device cuda:0 \
  --pdb_ref_dir /path/to/pdb_refs \
  --afdb_ref_dir /path/to/afdb_refs

Then run again for a second folder, e.g., checkpoint2. Compare the two summary.json files.
"""

import argparse
import json
import os
import sys
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

# Optional deps (import lazily where possible)
# biotite: structure parsing, Kabsch, TM-score fallback
# esm: ESMFold and ESM-IF

# -------------------------
# Utility & I/O
# -------------------------

# --- Stream sanitization: ensure tools see valid occ/B without modifying originals ---
from tempfile import NamedTemporaryFile

def _line_needs_patch(ln: str) -> bool:
    if not (ln.startswith("ATOM") or ln.startswith("HETATM")):
        return False
    s = ln.rstrip("
")
    if len(s) < 66:
        return True
    occ = s[54:60].strip()
    bfac = s[60:66].strip()
    try:
        float(occ); float(bfac)
        return False
    except Exception:
        return True


def ensure_readable_pdb(p: Path, occ: float = 1.00, bfac: float = 0.00) -> Tuple[Path, bool]:
    """Return (path_to_use, is_temp). If occ/B are missing or malformed, write a temp sanitized copy.
    Never mutates the original file.
    """
    needs = False
    with open(p, "r") as fin:
        for ln in fin:
            if _line_needs_patch(ln):
                needs = True
                break
    if not needs:
        return p, False

    tmp = NamedTemporaryFile("w", suffix=".pdb", delete=False)
    fmt_occ = f"{occ:6.2f}"
    fmt_b = f"{bfac:6.2f}"
    with open(p, "r") as fin:
        for ln in fin:
            if ln.startswith(("ATOM", "HETATM")):
                s = ln.rstrip("
").ljust(80)
                if _line_needs_patch(ln):
                    s = s[:54] + fmt_occ + fmt_b + s[66:]
                tmp.write(s[:80] + "
")
            else:
                tmp.write(ln)
    tmp.close()
    return Path(tmp.name), True

# --- Legacy on-disk patchers (kept for optional workflows) ---

# --- PDB patching: fill missing occupancy (1.00) and B-factor (0.00) ---
# (Optional utilities if you want to persist patched files on disk)

def patch_pdb_bfac_occupancy(in_path: Path, out_path: Path, occ: float = 1.00, bfac: float = 0.00):
    fmt_occ = f"{occ:6.2f}"
    fmt_bfac = f"{bfac:6.2f}"
    with open(in_path, "r") as fin, open(out_path, "w") as fout:
        for line in fin:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                s = line.rstrip("
").ljust(80)
                occ_field = s[54:60].strip()
                bfac_field = s[60:66].strip()
                def is_num(x):
                    try:
                        float(x); return True
                    except:
                        return False
                if not is_num(occ_field):
                    s = s[:54] + fmt_occ + s[60:]
                if not is_num(bfac_field):
                    s = s[:60] + fmt_bfac + s[66:]
                fout.write(s[:80] + "
")
            else:
                fout.write(line)

def preprocess_pdb_dir(input_dir: Path, out_dir: Path, occ: float = 1.00, bfac: float = 0.00) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in sorted(input_dir.glob("*.pdb")):
        patch_pdb_bfac_occupancy(p, out_dir / p.name, occ=occ, bfac=bfac)
    return out_dir

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


def list_pdbs(input_dir: Path) -> List[Path]:
    return sorted([p for p in input_dir.glob("*.pdb") if p.is_file()])


# -------------------------
# Structure helpers (Biotite)
# -------------------------

def load_atomarray_ca(pdb_path: Path):
    """Load PDB and extract CA AtomArray using biotite.
    Returns (array, residue_count)."""
    try:
        from biotite.structure.io.pdb import PDBFile
        from biotite.structure import AtomArray
    except Exception as e:
        raise RuntimeError("biotite is required for structure I/O. pip install biotite") from e

    pdb = PDBFile.read(str(pdb_path))
    arr = pdb.get_structure(model=1)
    ca = arr[arr.atom_name == "CA"]
    # Sort by chain+res_id to ensure consistent order
    order = np.lexsort((ca.res_id, ca.chain_id))
    ca = ca[order]
    return ca, int(len(ca))


def kabsch_align_and_rmsd(ref_ca, mob_ca) -> float:
    """Compute CA RMSD after Kabsch superposition using biotite."""
    from biotite.structure import superimpose
    # expect same length (ProteinMPNN/ESM-IF sequence matches backbone)
    if len(ref_ca) != len(mob_ca):
        m = min(len(ref_ca), len(mob_ca))
        ref_sel = ref_ca[:m]
        mob_sel = mob_ca[:m]
    else:
        ref_sel, mob_sel = ref_ca, mob_ca
    rot, trans = superimpose(mob_sel.coord, ref_sel.coord)
    sup = (mob_sel.coord @ rot.T) + trans
    diff2 = np.sum((sup - ref_sel.coord) ** 2, axis=1)
    return float(np.sqrt(np.mean(diff2)))


# -------------------------
# TM-score wrappers
# -------------------------

def run_tmalign(tmalign_path: Optional[str], pdb_a: Path, pdb_b: Path) -> Optional[float]:
    """If TMalign binary is available, run it and parse TM-score (normalized by A, the first input)."""
    exe = tmalign_path or "TMalign"
    try:
        out = subprocess.check_output([exe, str(pdb_a), str(pdb_b)], stderr=subprocess.STDOUT, text=True)
        # Parse line like: "TM-score=	0.61234 (if normalized by length of structure A)"
        for line in out.splitlines():
            if line.strip().startswith("TM-score=") and "length of structure A" in line:
                val = line.split("=")[-1].split()[0]
                return float(val)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    except Exception:
        return None
    return None


def tm_score_biotite(ca_a, ca_b) -> float:
    """Approximate TM-score via biotite utilities (without TMalign).
    Normalization by length of structure A to match TMalign behavior we parse above.
    """
    from biotite.structure import superimpose_structural_homologs, distance
    # Align subsets if lengths mismatch
    m = min(len(ca_a), len(ca_b))
    A = ca_a[:m]
    B = ca_b[:m]
    # Coarse homologous superposition (returns transformed B)
    B_sup = superimpose_structural_homologs(A, B)
    d = np.linalg.norm(A.coord - B_sup.coord, axis=1)
    # Standard TM-score parameters
    L_a = float(len(A))
    d0 = 1.24 * (L_a - 15) ** (1/3) - 1.8 if L_a > 15 else 0.5  # clamp small lengths
    d0 = max(d0, 0.5)
    score = np.sum(1.0 / (1.0 + (d / d0) ** 2)) / L_a
    return float(score)


def tm_score_pair(ca_a, ca_b, tmalign_path: Optional[str], tmp_dir: Path) -> float:
    """Try TMalign, fall back to biotite approximation."""
    # If using TMalign we need temporary PDBs for each CA subset
    if tmalign_path is not None or shutil_which("TMalign") is not None:
        pdb_a = write_temp_pdb_from_ca(ca_a, tmp_dir / "a.pdb")
        pdb_b = write_temp_pdb_from_ca(ca_b, tmp_dir / "b.pdb")
        val = run_tmalign(tmalign_path, pdb_a, pdb_b)
        if val is not None:
            return val
    # Fallback
    return tm_score_biotite(ca_a, ca_b)


def shutil_which(cmd: str) -> Optional[str]:
    from shutil import which
    return which(cmd)


def write_temp_pdb_from_ca(ca_arr, out_path: Path) -> Path:
    from biotite.structure.io.pdb import PDBFile
    pdb = PDBFile()
    # Write only CA atoms into a minimal PDB
    pdb.set_structure(ca_arr)
    pdb.write(str(out_path))
    return out_path


# -------------------------
# Sequence design: ProteinMPNN & ESM-IF
# -------------------------

def design_sequences_proteinmpnn(pdb_path: Path, n: int, proteinmpnn_repo: Optional[Path], device: str, workdir: Path) -> List[str]:
    """Call ProteinMPNN to design n sequences for the backbone. Uses the repo's inference script.
    Caches sequences in workdir/seqs/<basename>.fa
    """
    cache_fa = workdir / "seqs" / f"{pdb_path.stem}.fa"
    ensure_dir(cache_fa.parent)
    if cache_fa.exists():
        return [s for s in read_fasta_sequences(cache_fa)][:n]

    # Look for run_inference.py in the repo
    script = None
    if proteinmpnn_repo is not None:
        candidate = proteinmpnn_repo / "inference.py"
        if candidate.exists():
            script = candidate
        else:
            # Try common path
            for name in ["inference.py", "proteinmpnn_run.py", "run_inference.py"]:
                c = proteinmpnn_repo / name
                if c.exists():
                    script = c
                    break
    if script is None:
        # Try import-based API
        try:
            import proteinmpnn  # type: ignore
        except Exception as e:
            raise RuntimeError("ProteinMPNN not found. Provide --proteinmpnn_repo or install an importable package.") from e
        # Minimal import-based fallback (API varies; we attempt a generic route)
        # If this fails in your environment, prefer the CLI path above.
        try:
            model = proteinmpnn.ProteinMPNN(device=device)
            seqs = model.design(str(pdb_path), num_seqs=n)
            write_fasta(cache_fa, [(f">{pdb_path.stem}_{i}", s) for i, s in enumerate(seqs)])
            return seqs
        except Exception as e:
            raise RuntimeError("ProteinMPNN Python API call failed; please use --proteinmpnn_repo CLI path.") from e

    # CLI route
    out_dir = ensure_dir(workdir / "proteinmpnn_raw" / pdb_path.stem)
    cmd = [
        sys.executable,
        str(script),
        "--pdb",
        str(pdb_path),
        "--num_seq",
        str(n),
        "--out_folder",
        str(out_dir),
        "--device",
        device,
    ]
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ProteinMPNN CLI failed for {pdb_path}") from e

    # Find sequences in out_dir (collect first n)
    seqs = []
    for txt in sorted(out_dir.glob("*.fa")):
        seqs.extend(read_fasta_sequences(txt))
    if not seqs:
        # Some repos write .txt or .fasta
        for txt in sorted(out_dir.glob("*.txt")):
            seqs.extend(read_fasta_sequences(txt))
    seqs = seqs[:n]
    write_fasta(cache_fa, [(f">{pdb_path.stem}_{i}", s) for i, s in enumerate(seqs)])
    return seqs


def design_sequences_esm_if(pdb_path: Path, n: int, device: str, workdir: Path) -> List[str]:
    """Use ESM-IF1 inverse folding to design sequences. Requires esm>=2.0.
    Caches in workdir/seqs/<basename>_esmif.fa
    """
    cache_fa = workdir / "seqs" / f"{pdb_path.stem}_esmif.fa"
    ensure_dir(cache_fa.parent)
    if cache_fa.exists():
        return [s for s in read_fasta_sequences(cache_fa)][:n]

    try:
        import torch
        import esm
        from biotite.structure.io.pdb import PDBFile
    except Exception as e:
        raise RuntimeError("esm and biotite are required for ESM-IF. pip install esm biotite torch") from e

    pdb = load_pdb_string(pdb_path)

    # Load IF1
    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()  # type: ignore
    model = model.to(device)
    model.eval()

    # ESM-IF expects coords; we use utility from esm.inverse_folding
    from esm.inverse_folding.util import load_structure as esm_load_structure
    from esm.inverse_folding.util import \
        coord_batch_from_structure as esm_coord_batch_from_structure

    structure = esm_load_structure(str(pdb_path))
    batch = esm_coord_batch_from_structure(structure)

    seqs = []
    with torch.no_grad():
        for i in range(n):
            out = model.sample(**batch, temperature=1.0)
            seq = out["seqs"][0]
            seqs.append(seq)

    write_fasta(cache_fa, [(f">{pdb_path.stem}_esmif_{i}", s) for i, s in enumerate(seqs)])
    return seqs


# -------------------------
# ESMFold prediction
# -------------------------

def esmfold_predict(seq: str, device: str) -> Tuple[str, np.ndarray]:
    """Return PDB string and per-residue pLDDT array for a given AA sequence using esmfold_v1."""
    import torch
    import esm

    model = getattr(esm.pretrained, "esmfold_v1")()
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        output = model.infer_pdb(seq)
        # esmfold also produces confidence metrics; pLDDT is stored in B-factor field of PDB string
        # But we also compute plddt array from model if available
        try:
            # esmfold model can return output confidence via model outputs; however infer_pdb returns only PDB string.
            # We parse B-factors (pLDDT) from the PDB string.
            plddt = parse_plddt_from_pdb_string(output)
        except Exception:
            plddt = np.array([])
        return output, plddt


def parse_plddt_from_pdb_string(pdb_str: str) -> np.ndarray:
    """Extract B-factor column as pLDDT per residue (average over atoms in the residue)."""
    import re
    # PDB ATOM line: columns 61-66 = B-factor
    plddt_per_res: Dict[Tuple[str, int], List[float]] = {}
    for line in pdb_str.splitlines():
        if not line.startswith("ATOM"):
            continue
        chain = line[21].strip() or "A"
        res_id = int(line[22:26])
        try:
            b = float(line[60:66])
        except Exception:
            continue
        key = (chain, res_id)
        plddt_per_res.setdefault(key, []).append(b)
    if not plddt_per_res:
        return np.array([])
    res_means = [np.mean(v) for k, v in sorted(plddt_per_res.items(), key=lambda x: (x[0][0], x[0][1]))]
    return np.array(res_means)


def save_pdb_string(pdb_str: str, out_path: Path):
    with open(out_path, "w") as f:
        f.write(pdb_str)


def load_pdb_string(p: Path) -> str:
    with open(p, "r") as f:
        return f.read()


def read_fasta_sequences(path: Path) -> List[str]:
    seqs = []
    seq = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if seq:
                seqs.append("".join(seq))
                seq = []
        else:
            seq.append(line)
    if seq:
        seqs.append("".join(seq))
    return seqs


def write_fasta(path: Path, entries: List[Tuple[str, str]]):
    with open(path, "w") as f:
        for hdr, seq in entries:
            f.write(f"{hdr}\n")
            for i in range(0, len(seq), 80):
                f.write(seq[i:i+80] + "\n")


# -------------------------
# Core evaluation pipeline per design
# -------------------------

def evaluate_design(
    pdb_path: Path,
    n_sequences: int,
    device: str,
    seq_designer: str,
    proteinmpnn_repo: Optional[Path],
    esmfold_cache_dir: Path,
    tmalign_path: Optional[str],
) -> Dict:
    """Return dict with per-design results used downstream for metrics.
    Keys: name, len, designable(bool), min_sc_rmsd, best_seq_idx, mean_plddt
    """
    pdb_for_tools, tmp_created = ensure_readable_pdb(pdb_path)
    try:
        ca_gen, L = load_atomarray_ca(pdb_for_tools)
    finally:
        if tmp_created:
            try: os.unlink(pdb_for_tools)
            except: pass

    # 1) Inverse folding -> sequences
    if seq_designer == "proteinmpnn":
        try:
            seqs = design_sequences_proteinmpnn(pdb_for_tools, n_sequences, proteinmpnn_repo, device, esmfold_cache_dir)
        except Exception as e:
            print(f"[WARN] ProteinMPNN failed for {pdb_path.name}: {e}. Falling back to ESM-IF.")
            seqs = design_sequences_esm_if(pdb_for_tools, n_sequences, device, esmfold_cache_dir)
    elif seq_designer == "esm_if":
        seqs = design_sequences_esm_if(pdb_for_tools, n_sequences, device, esmfold_cache_dir)
    else:
        raise ValueError("seq_designer must be 'proteinmpnn' or 'esm_if'")

    # 2) ESMFold each sequence -> predicted structure, pLDDT
    pred_infos = []
    for i, seq in enumerate(seqs):
        out_pdb = esmfold_cache_dir / "esmfold" / f"{pdb_path.stem}_seq{i}.pdb"
        ensure_dir(out_pdb.parent)
        if out_pdb.exists():
            pdb_str = load_pdb_string(out_pdb)
            plddt = parse_plddt_from_pdb_string(pdb_str)
        else:
            pdb_str, plddt = esmfold_predict(seq, device)
            save_pdb_string(pdb_str, out_pdb)
        # Extract CA
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdb")
        tmp.write(pdb_str.encode("utf-8"))
        tmp.flush()
        ca_pred, _ = load_atomarray_ca(Path(tmp.name))
        os.unlink(tmp.name)
        pred_infos.append({"seq_idx": i, "ca": ca_pred, "plddt": plddt, "mean_plddt": float(np.mean(plddt)) if plddt.size>0 else float("nan")})

    # 3) Select most similar prediction by TM-score (paper step) and compute scRMSD against it
    best_idx = None
    best_tm = -1.0
    best_rmsd = float("inf")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for info in pred_infos:
            tm = tm_score_pair(ca_gen, info["ca"], tmalign_path, td)
            if tm > best_tm:
                best_tm = tm
                rmsd = kabsch_align_and_rmsd(ca_gen, info["ca"])
                best_rmsd = rmsd
                best_idx = info["seq_idx"]

    mean_plddt_best = pred_infos[best_idx]["mean_plddt"] if best_idx is not None else float("nan")
    designable = (best_rmsd <= 2.0) and (mean_plddt_best >= 70.0)

    return {
        "name": pdb_path.stem,
        "length": L,
        "designable": bool(designable),
        "min_sc_rmsd": float(best_rmsd),
        "best_seq_idx": int(best_idx if best_idx is not None else -1),
        "mean_plddt": float(mean_plddt_best),
    }


# -------------------------
# Diversity and Novelty clustering
# -------------------------

def cluster_by_tm_threshold(ca_list: List, threshold: float, tmalign_path: Optional[str]) -> Tuple[int, List[int]]:
    """Build graph connecting pairs with TM >= threshold and return (#clusters, labels).
    labels[i] is the cluster id of i-th CA in the input list.
    """
    import networkx as nx
    G = nx.Graph()
    n = len(ca_list)
    G.add_nodes_from(range(n))
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for i in range(n):
            for j in range(i+1, n):
                tm = tm_score_pair(ca_list[i], ca_list[j], tmalign_path, td)
                if tm >= threshold:
                    G.add_edge(i, j)
    comps = list(nx.connected_components(G))
    labels = [None]*n
    for c_id, comp in enumerate(comps):
        for idx in comp:
            labels[idx] = c_id
    return len(comps), labels


def compute_diversity(designable_mask: np.ndarray, ca_list: List, tmalign_path: Optional[str]) -> Tuple[float, int, List[int]]:
    """Return diversity fraction (clusters/N), number of clusters, and labels for designables only."""
    idxs = [i for i, d in enumerate(designable_mask) if d]
    if not idxs:
        return 0.0, 0, []
    sub_cas = [ca_list[i] for i in idxs]
    k, labels_sub = cluster_by_tm_threshold(sub_cas, threshold=0.6, tmalign_path=tmalign_path)
    # Map back to full list
    labels_full = [-1]*len(ca_list)
    for i_sub, i_full in enumerate(idxs):
        labels_full[i_full] = labels_sub[i_sub]
    diversity_fraction = k / float(len(ca_list))
    return diversity_fraction, k, labels_full


def compute_novelty(
    designable_mask: np.ndarray,
    ca_list: List,
    ref_dirs: List[Path],
    tmalign_path: Optional[str],
) -> Tuple[float, int]:
    """Novelty fraction = #novel_clusters / N, where a design is novel if designable AND
    TM <= 0.5 to *every* structure in reference sets.
    Clustering of the novel subset uses TM >= 0.6 single-linkage.
    """
    if not ref_dirs:
        return 0.0, 0

    novel_idxs = []
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for i, ca in enumerate(ca_list):
            if not designable_mask[i]:
                continue
            is_novel = True
            for ref_dir in ref_dirs:
                for ref_pdb in ref_dir.glob("*.pdb"):
                    ref_ca, _ = load_atomarray_ca(ref_pdb)
                    tm = tm_score_pair(ca, ref_ca, tmalign_path, td)
                    if tm > 0.5:
                        is_novel = False
                        break
                if not is_novel:
                    break
            if is_novel:
                novel_idxs.append(i)

    if not novel_idxs:
        return 0.0, 0

    sub_cas = [ca_list[i] for i in novel_idxs]
    k, _ = cluster_by_tm_threshold(sub_cas, threshold=0.6, tmalign_path=tmalign_path)
    novelty_fraction = k / float(len(ca_list))
    return novelty_fraction, k


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser(description="Evaluate Genie 2-style metrics for a folder of PDBs")
    ap.add_argument("--input_dir", type=Path, required=True, help="Folder of generated PDBs")
    ap.add_argument("--workdir", type=Path, required=True, help="Working directory for caches & outputs")
    ap.add_argument("--seq_designer", type=str, default="proteinmpnn", choices=["proteinmpnn", "esm_if"], help="Inverse folding backend")
    ap.add_argument("--proteinmpnn_repo", type=Path, default=None, help="Path to ProteinMPNN repo (for CLI)")
    ap.add_argument("--device", type=str, default="cuda:0", help="Torch device for ESMFold/ESM-IF")
    ap.add_argument("--n_sequences", type=int, default=8, help="# sequences per design (paper uses 8)")
    ap.add_argument("--tmalign", type=str, default=None, help="Path to TMalign binary (optional; else biotite fallback)")
    ap.add_argument("--pdb_ref_dir", type=Path, default=None, help="Directory of PDB reference structures (optional)")
    ap.add_argument("--afdb_ref_dir", type=Path, default=None, help="Directory of AFDB reference structures (optional)")
        args = ap.parse_args()

    input_dir: Path = args.input_dir
    workdir: Path = ensure_dir(args.workdir)
    seq_designer: str = args.seq_designer
    proteinmpnn_repo: Optional[Path] = args.proteinmpnn_repo
    device: str = args.device
    n_sequences: int = args.n_sequences
    tmalign_path: Optional[str] = args.tmalign
    pdb_paths = list_pdbs(input_dir)
    if not pdb_paths:
        print(f"No PDBs found in {input_dir}")
        sys.exit(1)

    per_sample_rows = []
    ca_list = []
    designable_mask = []

    # Evaluate per design
    for pdb_path in tqdm(pdb_paths, desc="Evaluating designs"):
        try:
            res = evaluate_design(
                pdb_path=pdb_path,
                n_sequences=n_sequences,
                device=device,
                seq_designer=seq_designer,
                proteinmpnn_repo=proteinmpnn_repo,
                esmfold_cache_dir=workdir,
                tmalign_path=tmalign_path,
            )
        except Exception as e:
            print(f"[ERROR] {pdb_path.name}: {e}")
            res = {
                "name": pdb_path.stem,
                "length": np.nan,
                "designable": False,
                "min_sc_rmsd": np.inf,
                "best_seq_idx": -1,
                "mean_plddt": np.nan,
            }
        per_sample_rows.append(res)
        # Also cache CA array for clustering (sanitize on the fly)
        try:
            p_use, was_tmp = ensure_readable_pdb(pdb_path)
            try:
                ca, _ = load_atomarray_ca(p_use)
            finally:
                if was_tmp:
                    try: os.unlink(p_use)
                    except: pass
        except Exception:
            ca = None
        ca_list.append(ca)
        designable_mask.append(bool(res["designable"]))

    # Compute metrics
    N = len(pdb_paths)
    pstructures = float(np.mean(designable_mask))  # fraction designable

    # Diversity (clusters among designables)
    ca_valid = [c for c in ca_list]
    diversity_fraction, n_clusters, labels = compute_diversity(np.array(designable_mask, dtype=bool), ca_valid, tmalign_path)

    # F1 (beta=1)
    if (pstructures + diversity_fraction) == 0:
        F1 = 0.0
    else:
        F1 = 2 * pstructures * diversity_fraction / (pstructures + diversity_fraction)

    # Novelty
    novelty_pdb = 0.0
    novelty_pdb_k = 0
    novelty_afdb = 0.0
    novelty_afdb_k = 0

    if args.pdb_ref_dir and args.pdb_ref_dir.exists():
        novelty_pdb, novelty_pdb_k = compute_novelty(np.array(designable_mask, dtype=bool), ca_valid, [args.pdb_ref_dir], tmalign_path)
    if args.afdb_ref_dir and args.afdb_ref_dir.exists():
        novelty_afdb, novelty_afdb_k = compute_novelty(np.array(designable_mask, dtype=bool), ca_valid, [args.afdb_ref_dir], tmalign_path)

    # Save outputs
    out_dir = ensure_dir(workdir)
    pd.DataFrame(per_sample_rows).to_csv(out_dir / "per_sample.csv", index=False)

    summary = {
        "N": N,
        "pstructures": pstructures,
        "pclusters": diversity_fraction,
        "F1": F1,
        "diversity_num_clusters": n_clusters,
        "novelty_pdb": novelty_pdb,
        "novelty_pdb_num_clusters": novelty_pdb_k,
        "novelty_afdb": novelty_afdb,
        "novelty_afdb_num_clusters": novelty_afdb_k,
        "thresholds": {"scRMSD_max": 2.0, "mean_pLDDT_min": 70.0, "tm_cluster": 0.6, "tm_novel_max": 0.5},
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
