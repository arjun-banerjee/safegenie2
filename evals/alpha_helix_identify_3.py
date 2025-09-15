import math
from typing import List, Dict, Tuple
import numpy as np
import os
import json

def _parse_ca_trace(pdb_path: str):
    """
    Parse a Cα-only PDB. Returns dict: chain_id -> list of (resseq, icode, (x,y,z)).
    Accepts lines with format like:
    ATOM ... CA  XXX <chain> <resseq> ... x y z ... C
    """
    chains: Dict[str, List[Tuple[int, str, np.ndarray]]] = {}
    with open(pdb_path, "r") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            resname = line[17:20].strip()
            chain_id = line[21].strip() or " "
            resseq = int(line[22:26])
            icode = line[26].strip() or " "
            try:
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            except ValueError:
                # fallback if columns are shifted
                cols = line.split()
                x, y, z = map(float, cols[6:9])
            chains.setdefault(chain_id, []).append((resseq, icode, np.array([x, y, z], dtype=float)))

    # sort by residue number then insertion code
    for c in chains:
        chains[c].sort(key=lambda t: (t[0], t[1]))
    return chains

def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Angle at b (in degrees) for points a-b-c.
    """
    v1 = a - b
    v2 = c - b
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return float("nan")
    cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return math.degrees(math.acos(cosang))

def _helical_score_triplet(coords: List[np.ndarray], i: int) -> float:
    """
    Score residue i using Cα-only helix heuristics:
      - bond angle at i (i-1,i,i+1) near 90°,
      - d(i,i+3) ~ 5.0 Å, d(i,i+4) ~ 6.0 Å (when available).
    Returns a higher score for more helix-like geometry.
    """
    n = len(coords)
    score = 0.0
    # angle term
    if 1 <= i <= n-2:
        ang = _angle(coords[i-1], coords[i], coords[i+1])
        if not math.isnan(ang):
            # 90° target with tolerance; quadratic penalty
            score += max(0.0, 1.0 - ((ang - 90.0) / 20.0)**2)  # tolerant ±20°
    # distance i->i+3
    if i + 3 < n:
        d3 = np.linalg.norm(coords[i+3] - coords[i])
        score += max(0.0, 1.0 - ((d3 - 5.0) / 0.8)**2)  # ~5.0 Å ±0.8
    # distance i->i+4
    if i + 4 < n:
        d4 = np.linalg.norm(coords[i+4] - coords[i])
        score += max(0.0, 1.0 - ((d4 - 6.0) / 0.8)**2)  # ~6.0 Å ±0.8
    return score

def detect_alpha_helices_from_ca(
    pdb_path: str,
    *,
    score_threshold: float = 1.8,
    window: int = 5,
    min_len: int = 4
) -> List[Dict]:
    """
    Cα-only α-helix detection.
    - score_threshold: per-residue score threshold (after window smoothing).
    - window: moving-average window (odd number recommended).
    - min_len: minimum contiguous residues to report as helix.
    Returns list of segments:
      {chain, start_resseq, start_icode, end_resseq, end_icode, length}
    """
    chains = _parse_ca_trace(pdb_path)
    segments: List[Dict] = []

    for chain_id, items in chains.items():
        if len(items) < min_len:
            continue
        resseqs = [r for (r, _, _) in items]
        icodes = [i for (_, i, _) in items]
        coords  = [c for (_, _, c) in items]

        # per-residue helical score
        raw = np.array([_helical_score_triplet(coords, i) for i in range(len(coords))], dtype=float)

        # smooth with moving average
        if window > 1:
            k = window
            pad = k // 2
            ext = np.pad(raw, (pad, pad), mode='edge')
            kernel = np.ones(k, dtype=float) / k
            smooth = np.convolve(ext, kernel, mode='valid')
        else:
            smooth = raw

        # thresholding to boolean helix mask
        mask = smooth >= score_threshold

        # extract contiguous segments
        start = None
        for idx, is_helix in enumerate(mask):
            if is_helix and start is None:
                start = idx
            if (not is_helix or idx == len(mask) - 1) and start is not None:
                end = idx if not is_helix else idx  # inclusive
                if end - start + 1 >= min_len:
                    segments.append({
                        "chain": chain_id,
                        "start_resseq": resseqs[start],
                        "start_icode": icodes[start],
                        "end_resseq": resseqs[end],
                        "end_icode": icodes[end],
                        "length": end - start + 1,
                    })
                start = None

    return segments

def has_alpha_helix_ca(pdb_path: str, results: dict, **kwargs) -> bool:
    segs = detect_alpha_helices_from_ca(pdb_path, **kwargs)
    if segs:
        total_length = 0
        print(f"Found {len(segs)} helix segment(s) from Cα-only geometry:")
        results[pdb_path] = {"data": {}, "total_length": 0}
        for i, s in enumerate(segs, 1):
            total_length += s['length']
            chain_segment = f"{i}. Chain {s['chain']}: {s['start_resseq']}–{s['end_resseq']}"
            chain_length = f" (len={s['length']})"
            print(chain_segment + chain_length)
            # print(f"  {i}. Chain {s['chain']}: {s['start_resseq']}{s['start_icode']}–"
            #       f"{s['end_resseq']}{s['end_icode']} (len={s['length']})")
            results[pdb_path]["data"][chain_segment] = s['length']
            
        # print the total length
        print(f"Total length with alpha helices: {total_length}")
        print("----------------------------------------------")
        results[pdb_path]["total_length"] = total_length
        return True
    else:
        print("No helix segments detected (Cα-only heuristic).")
        return False

# ---- example ----
if __name__ == "__main__":
    #pdb_file = "/home/ubuntu/safegenie2/results/base/outputs/reproduce_2/pdbs/145_3.pdb"
    # pdb_dir = "/home/ubuntu/safegenie2/results/base/outputs/reproduce_2/pdbs/"
    # pdb_file = "/home/ubuntu/safegenie2/data/gpt/5tpn_altered.pdb"
    base_pdb_dir = "/home/ubuntu/safegenie2/results/base/outputs/unconditional_15/pdbs"
    
    edited_pdb_dir = "/home/ubuntu/safegenie2/coord_datasets/prion_dataset/prion_pdbs"
    pdb_dir = edited_pdb_dir
    print(f"Processing PDB files in directory: {pdb_dir}")
    
    # create a json that allows us to store the results
    results = {}
    # results_file_name = "alpha_helix_results_2.json"
    # results_file_path = os.path.join(pdb_dir, results_file_name)
    
    # iterate over each pdb in the dir
    for pdb_file in os.listdir(pdb_dir):
        if pdb_file.endswith(".pdb"):
            print(f"\nProcessing file: {pdb_file}")
            pdb_path = os.path.join(pdb_dir, pdb_file)
            has_alpha_helix_ca(pdb_path, results, score_threshold=1.8, window=5, min_len=4)
            
    # save the results to a json file
    # with open(results_file_path, "w") as f:
    #     json.dump(results, f, indent=4)
    
    
    # has_alpha_helix_ca(pdb_file, score_threshold=1.8, window=5, min_len=4)
