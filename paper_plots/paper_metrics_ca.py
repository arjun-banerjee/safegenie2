#!/usr/bin/env python3
"""
Paper-faithful metrics for Cα-only ALA PDBs.

Features:
- Optionally compute scTM (self-consistency TM) via TM-align between each generated PDB
  and its ESMFold-predicted PDB with the same basename; writes info.csv.
- Designability pstructures = #designable / N  (designable if scTM >= --sctm_thresh)
- Diversity pclusters = #single-linkage clusters among designables / N, with edges if TM >= --tm_cluster
- F1 (β=1) = harmonic mean of pstructures and pclusters
- Novelty (optional): among designables, keep those with max TM to any reference <= --tm_novel,
  then cluster at --tm_cluster and report (#novel clusters)/N.
- Per-length (exact Cα count) breakdown of pstructures, pclusters, and F1.
- Optional per-file helix% / strand% from Cα-only heuristics.

Requires:
- numpy
- TMalign binary in PATH (https://zhanggroup.org/TM-align/)
"""

import os, sys, csv, json, glob, math, shutil, argparse, subprocess
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional
import numpy as np

# ------------------------ TM-align helpers ------------------------

def require_tmalign() -> str:
    exe = shutil.which("TMalign")
    if not exe:
        raise FileNotFoundError(
            "TMalign not found in PATH. Install it and ensure `TMalign` is runnable."
        )
    return exe

def tm_align(pdb_a: str, pdb_b: str) -> float:
    """
    Returns the first 'TM-score' value printed by TMalign.
    Works for Cα-only structures.
    """
    exe = require_tmalign()
    p = subprocess.run([exe, pdb_a, pdb_b], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out = (p.stdout or "") + "\n" + (p.stderr or "")
    for line in out.splitlines():
        if "TM-score" in line:
            parts = line.replace("=", " ").split()
            for tok in parts:
                try:
                    return float(tok)
                except:
                    pass
    raise RuntimeError(f"Could not parse TM-score for {os.path.basename(pdb_a)} vs {os.path.basename(pdb_b)}")

# ------------------------ scTM I/O & computation ------------------------

def load_sctm_csv(path: str) -> Dict[str, float]:
    m = {}
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            name = row.get("file") or row.get("pdb") or row.get("name") or row.get("basename")
            val  = row.get("scTM") or row.get("sctm") or row.get("self_consistency_tm") or row.get("tm")
            if name and val:
                try:
                    m[os.path.basename(name)] = float(val)
                except:
                    pass
    return m

def load_sctm_json(path: str) -> Dict[str, float]:
    data = json.load(open(path))
    m = {}
    if isinstance(data, dict):
        for k, v in data.items():
            try:
                m[os.path.basename(k)] = float(v)
            except:
                pass
    elif isinstance(data, list):
        for row in data:
            name = row.get("file") or row.get("pdb") or row.get("name")
            val  = row.get("scTM") or row.get("sctm") or row.get("tm")
            if name and val:
                try:
                    m[os.path.basename(name)] = float(val)
                except:
                    pass
    return m

def compute_sctm_from_preds(gen_file: str, pred_dir: str) -> Optional[float]:
    base = os.path.splitext(os.path.basename(gen_file))[0]
    cand = os.path.join(pred_dir, base + ".pdb")
    if not os.path.isfile(cand):
        return None
    try:
        return tm_align(gen_file, cand)
    except Exception:
        return None

def maybe_write_info_csv(out_csv_path: str, sctm_map: Dict[str, float]) -> None:
    # Write as: file,scTM
    with open(out_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "scTM"])
        for k, v in sorted(sctm_map.items()):
            w.writerow([k, f"{v:.6f}"])

# ------------------------ Single-linkage clustering via graph (TM >= threshold) ------------------------

def single_linkage_clusters(files: List[str], tm_thresh: float) -> Tuple[int, List[int]]:
    """
    Build an undirected graph with edge (i,j) if TM(i,j) >= tm_thresh.
    Connected components are the single-linkage clusters.
    Returns (num_clusters, labels[0..n-1]).
    """
    n = len(files)
    if n == 0:
        return 0, []
    if n == 1:
        return 1, [0]

    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            tm = tm_align(files[i], files[j])
            if tm >= tm_thresh:
                adj[i].append(j)
                adj[j].append(i)

    labels = [-1] * n
    k = 0
    for s in range(n):
        if labels[s] != -1:
            continue
        q = deque([s]); labels[s] = k
        while q:
            u = q.popleft()
            for v in adj[u]:
                if labels[v] == -1:
                    labels[v] = k
                    q.append(v)
        k += 1
    return k, labels

# ------------------------ Cα-only parsing + optional helix/strand percentages ------------------------

def parse_ca_coords(pdb_path: str) -> np.ndarray:
    coords = []
    with open(pdb_path, "r") as f:
        for line in f:
            if not line.startswith("ATOM"): continue
            if line[12:16].strip() != "CA": continue
            try:
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            except ValueError:
                cols = line.split()
                x, y, z = map(float, cols[6:9])
            coords.append((x, y, z))
    if not coords:
        raise ValueError(f"No CA atoms in {pdb_path}")
    return np.array(coords, dtype=float)

def ca_length(pdb_path: str) -> int:
    n = 0
    with open(pdb_path, "r") as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                n += 1
    return n

def angle(a,b,c):
    v1=a-b; v2=c-b
    n1=np.linalg.norm(v1); n2=np.linalg.norm(v2)
    if n1<1e-6 or n2<1e-6: return float("nan")
    return math.degrees(math.acos(np.clip(np.dot(v1,v2)/(n1*n2), -1.0, 1.0)))

def dihedral(p0,p1,p2,p3):
    b0=p1-p0; b1=p2-p1; b2=p3-p2
    b1n=b1/(np.linalg.norm(b1)+1e-8)
    v=b0-np.dot(b0,b1n)*b1n
    w=b2-np.dot(b2,b1n)*b1n
    x=np.dot(v,w); y=np.dot(np.cross(b1n,v), w)
    return math.degrees(math.atan2(y,x))

def helical_score_triplet(C: np.ndarray, i: int) -> float:
    n=len(C); s=0.0
    if 1<=i<=n-2:
        ang=angle(C[i-1],C[i],C[i+1])
        if not math.isnan(ang): s += max(0.0, 1.0 - ((ang-90.0)/20.0)**2)
    if i+3<n:
        d3=np.linalg.norm(C[i+3]-C[i]); s += max(0.0, 1.0 - ((d3-5.0)/0.8)**2)
    if i+4<n:
        d4=np.linalg.norm(C[i+4]-C[i]); s += max(0.0, 1.0 - ((d4-6.0)/0.8)**2)
    return s

def beta_score_quad(C: np.ndarray, i: int) -> float:
    n=len(C); s=0.0
    if 1<=i<=n-2:
        ang=angle(C[i-1],C[i],C[i+1])
        if not math.isnan(ang): s += max(0.0, 1.0 - ((ang-125.0)/20.0)**2)
    if i+2<n:
        d2=np.linalg.norm(C[i+2]-C[i]); s += max(0.0, 1.0 - ((d2-6.7)/0.8)**2)
    if 0<=i-1 and i+2<n:
        dih=abs(dihedral(C[i-1],C[i],C[i+1],C[i+2]))
        s += max(0.0, 1.0 - ((min(dih,360-dih)-180.0)/30.0)**2)
    return s

def smooth_mask(scores: np.ndarray, thr: float, win: int) -> np.ndarray:
    if win>1:
        k=win; pad=k//2
        ext=np.pad(scores,(pad,pad),mode='edge')
        sm=np.convolve(ext, np.ones(k)/k, mode='valid')
    else:
        sm=scores
    return sm>=thr

def secondary_percentages(C: np.ndarray,
                          helix_thr=1.8, helix_win=5,
                          beta_thr=1.6,  beta_win=5) -> Tuple[float,float]:
    n=len(C)
    if n==0: return 0.0, 0.0
    H = np.array([helical_score_triplet(C,i) for i in range(n)], float)
    E = np.array([beta_score_quad(C,i)     for i in range(n)], float)
    Hm = smooth_mask(H, helix_thr, helix_win)
    Em = smooth_mask(E, beta_thr,  beta_win)
    # resolve overlaps conservatively
    overlap = Hm & Em
    if overlap.any():
        if Hm.sum() <= Em.sum(): Hm[overlap] = False
        else:                    Em[overlap] = False
    return 100.0*Hm.sum()/n, 100.0*Em.sum()/n

# ------------------------ Paper-faithful metrics ------------------------

def compute_metrics(
    gen_dir: str,
    *,
    sctm_map: Dict[str, float],
    sctm_thresh: float = 0.5,
    tm_cluster: float = 0.6,
    ref_dir: Optional[str] = None,
    tm_novel_max: float = 0.5,
    do_per_length: bool = True,
    do_secondary: bool = True,
    helix_thr: float = 1.8, helix_win: int = 5,
    beta_thr: float = 1.6,  beta_win: int = 5,
):
    files = sorted(glob.glob(os.path.join(gen_dir, "*.pdb")))
    if not files:
        raise RuntimeError("No PDBs found in --gen_dir")

    # per-file basics & optional secondary structure
    per_file = []
    length_groups = defaultdict(list)
    for fp in files:
        name = os.path.basename(fp)
        L = ca_length(fp)
        rec = {"file": name, "path": fp, "length": L}
        if do_secondary:
            C = parse_ca_coords(fp)
            h, e = secondary_percentages(C, helix_thr, helix_win, beta_thr, beta_win)
            rec["pct_helix"] = h
            rec["pct_strand"] = e
        per_file.append(rec)
        length_groups[L].append(fp)

    # mark designable from scTM
    N = len(files)
    designable = []
    for rec in per_file:
        name = rec["file"]
        sc = sctm_map.get(name)
        rec["scTM"] = sc
        rec["designable"] = (sc is not None and sc >= sctm_thresh)
        if rec["designable"]:
            designable.append(rec["path"])

    pstructures = len(designable) / N

    # diversity (single linkage among designables, TM >= tm_cluster)
    if len(designable) == 0:
        num_clusters = 0
    elif len(designable) == 1:
        num_clusters = 1
    else:
        num_clusters, _ = single_linkage_clusters(designable, tm_cluster)
    pclusters = num_clusters / N

    F1 = 0.0 if (pstructures + pclusters) == 0 else (2 * pstructures * pclusters) / (pstructures + pclusters)

    # novelty (optional)
    novelty = None
    if ref_dir:
        ref_files = sorted(glob.glob(os.path.join(ref_dir, "*.pdb")))
        if not ref_files:
            raise RuntimeError("No PDBs found in --ref_dir")
        novel = []
        for f in designable:
            max_tm = 0.0
            for r in ref_files:
                tm = tm_align(f, r)
                if tm > max_tm:
                    max_tm = tm
                if max_tm > tm_novel_max:
                    break
            if max_tm <= tm_novel_max:
                novel.append(f)
        if len(novel) == 0:
            novelty = 0.0
        elif len(novel) == 1:
            novelty = 1.0 / N
        else:
            nclust, _ = single_linkage_clusters(novel, tm_cluster)
            novelty = nclust / N

    # per-length metrics (paper Fig. 6 style)
    per_length = []
    if do_per_length:
        for L, group in sorted(length_groups.items()):
            N_L = len(group)
            des_L = [f for f in group if sctm_map.get(os.path.basename(f), -1) >= sctm_thresh]
            pstruct_L = len(des_L) / N_L
            if len(des_L) <= 1:
                clusters_L = len(des_L)
            else:
                clusters_L, _ = single_linkage_clusters(des_L, tm_cluster)
            pclust_L = clusters_L / N_L
            F1_L = 0.0 if (pstruct_L + pclust_L) == 0 else (2 * pstruct_L * pclust_L) / (pstruct_L + pclust_L)
            per_length.append({
                "length": L, "N": N_L,
                "pstructures": pstruct_L,
                "pclusters":   pclust_L,
                "F1":          F1_L
            })

    return {
        "N": N,
        "designable": sum(1 for r in per_file if r["designable"]),
        "pstructures": pstructures,
        "num_designable_clusters": num_clusters,
        "pclusters": pclusters,
        "F1": F1,
        "novelty": novelty,
        "per_file": per_file,
        "per_length": per_length,
    }

# ------------------------ CLI ------------------------

def main():
    ap = argparse.ArgumentParser(description="Paper-faithful diversity/designability/F1 for Cα-only PDBs (with optional novelty)")
    ap.add_argument("--gen_dir", required=True, help="Directory of generated *.pdb")
    # scTM sources:
    ap.add_argument("--sctm_csv", help="CSV with columns [file, scTM] (alt headers tolerated)")
    ap.add_argument("--sctm_json", help="JSON mapping file->scTM")
    ap.add_argument("--pred_dir", help="Directory of ESMFold predictions with matching basenames to compute scTM")
    ap.add_argument("--write_info_csv", action="store_true", help="Write info.csv with computed/merged scTM to gen_dir")
    # thresholds:
    ap.add_argument("--sctm_thresh", type=float, default=0.5, help="Designable if scTM >= this")
    ap.add_argument("--tm_cluster",  type=float, default=0.6, help="TM threshold for single-linkage clustering")
    # novelty:
    ap.add_argument("--ref_dir", help="Reference PDB dir for novelty (optional)")
    ap.add_argument("--tm_novel", type=float, default=0.5, help="Novel if max TM to any reference <= this")
    # secondary structure reporting:
    ap.add_argument("--no_secondary", action="store_true", help="Skip helix/strand percentage computation")
    ap.add_argument("--helix_thr", type=float, default=1.8)
    ap.add_argument("--helix_win", type=int,   default=5)
    ap.add_argument("--beta_thr",  type=float, default=1.6)
    ap.add_argument("--beta_win",  type=int,   default=5)
    # outputs:
    ap.add_argument("--out_prefix", default="paper_metrics", help="Prefix for output files in gen_dir")
    args = ap.parse_args()

    gen_dir = args.gen_dir
    files = sorted(glob.glob(os.path.join(gen_dir, "*.pdb")))
    if not files:
        print("No PDBs found in --gen_dir", file=sys.stderr)
        sys.exit(1)

    # Build scTM map from provided sources
    sctm_map: Dict[str, float] = {}
    if args.sctm_csv:
        sctm_map.update(load_sctm_csv(args.sctm_csv))
    if args.sctm_json:
        sctm_map.update(load_sctm_json(args.sctm_json))

    # Optionally compute missing scTM via predictions and write info.csv
    if args.pred_dir:
        require_tmalign()  # fail fast if missing
        for fp in files:
            name = os.path.basename(fp)
            if name not in sctm_map:
                val = compute_sctm_from_preds(fp, args.pred_dir)
                if val is not None:
                    sctm_map[name] = val

    # Optionally write info.csv to gen_dir
    if args.write_info_csv:
        out_csv = os.path.join(gen_dir, "info.csv")
        maybe_write_info_csv(out_csv, sctm_map)
        print(f"[OK] wrote {out_csv}")

    # Compute metrics
    results = compute_metrics(
        gen_dir,
        sctm_map=sctm_map,
        sctm_thresh=args.sctm_thresh,
        tm_cluster=args.tm_cluster,
        ref_dir=args.ref_dir,
        tm_novel_max=args.tm_novel,
        do_per_length=True,
        do_secondary=(not args.no_secondary),
        helix_thr=args.helix_thr, helix_win=args.helix_win,
        beta_thr=args.beta_thr,   beta_win=args.beta_win
    )

    # Write outputs
    base = os.path.join(gen_dir, args.out_prefix)
    # summary
    summary = {
        "N": results["N"],
        "designable": results["designable"],
        "pstructures": results["pstructures"],
        "num_designable_clusters": results["num_designable_clusters"],
        "pclusters": results["pclusters"],
        "F1": results["F1"],
        "tm_cluster_threshold": args.tm_cluster,
        "sctm_threshold": args.sctm_thresh
    }
    if results["novelty"] is not None:
        summary["novelty"] = results["novelty"]
        summary["tm_novel_threshold"] = args.tm_novel

    with open(base + "_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # per-file (includes optional secondary)
    with open(base + "_per_file.json", "w") as f:
        json.dump(results["per_file"], f, indent=2)

    # per-length CSV
    with open(base + "_per_length.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["length","N","pstructures","pclusters","F1"])
        for r in results["per_length"]:
            w.writerow([r["length"], r["N"],
                        f"{r['pstructures']:.6f}",
                        f"{r['pclusters']:.6f}",
                        f"{r['F1']:.6f}"])

    print("[OK] wrote:")
    print(" ", base + "_summary.json")
    print(" ", base + "_per_file.json")
    print(" ", base + "_per_length.csv")

if __name__ == "__main__":
    main()
