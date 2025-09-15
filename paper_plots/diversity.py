import os, glob, csv, json, subprocess, shutil
from collections import defaultdict, deque

# -------------------- TM-align helpers --------------------
def _require_tmalign() -> str:
    exe = shutil.which("TMalign")
    if not exe:
        raise FileNotFoundError("TMalign not found in PATH.")
    return exe

def _tm_align(a: str, b: str) -> float:
    exe = _require_tmalign()
    out = subprocess.run([exe, a, b], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    text = (out.stdout or "") + "\n" + (out.stderr or "")
    for line in text.splitlines():
        if "TM-score" in line:
            parts = line.replace("=", " ").split()
            for tok in parts:
                try:
                    return float(tok)
                except:
                    pass
    raise RuntimeError(f"TM-score parse failed for {os.path.basename(a)} vs {os.path.basename(b)}")

# -------------------- graph single-linkage (TM ≥ 0.6) --------------------
def _clusters_single_linkage(files, tm_thresh=0.6):
    """
    Build an undirected graph with an edge (i,j) iff TM(i,j) ≥ tm_thresh.
    Connected components are single-linkage clusters.
    Returns (num_clusters, labels[list[int]]) in 0..K-1; labels indexed as files.
    """
    n = len(files)
    if n == 0:
        return 0, []
    if n == 1:
        return 1, [0]

    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            tm = _tm_align(files[i], files[j])
            if tm >= tm_thresh:
                adj[i].append(j)
                adj[j].append(i)

    labels = [-1]*n
    k = 0
    for s in range(n):
        if labels[s] != -1: 
            continue
        # BFS/DFS for component
        q = deque([s]); labels[s] = k
        while q:
            u = q.popleft()
            for v in adj[u]:
                if labels[v] == -1:
                    labels[v] = k
                    q.append(v)
        k += 1
    return k, labels

# -------------------- scTM ingestion --------------------
def load_sctm_csv(csv_path):
    m = {}
    with open(csv_path) as f:
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

def load_sctm_json(json_path):
    m = {}
    data = json.load(open(json_path))
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

# -------------------- paper-faithful metrics --------------------
def compute_paper_metrics(
    gen_dir: str,
    *,
    sctm_csv: str = None,
    sctm_json: str = None,
    sctm_thresh: float = 0.5,
    tm_cluster: float = 0.6,
    ref_dir: str = None,          # for novelty
    tm_novel_max: float = 0.5
):
    # All generated files (universe N)
    gen_files = sorted(glob.glob(os.path.join(gen_dir, "*.pdb")))
    N = len(gen_files)
    if N == 0:
        raise RuntimeError("No PDBs found in generated directory.")

    # Load scTM map
    sctm = {}
    if sctm_csv:  sctm.update(load_sctm_csv(sctm_csv))
    if sctm_json: sctm.update(load_sctm_json(sctm_json))

    # Keep designable subset (scTM ≥ threshold)
    designable = []
    for f in gen_files:
        base = os.path.basename(f)
        sc = sctm.get(base, None)
        if sc is not None and sc >= sctm_thresh:
            designable.append(f)

    # pstructures
    pstructures = len(designable) / N

    # Diversity: cluster ONLY the designable set by TM ≥ 0.6 (single linkage)
    if len(designable) == 0:
        num_clusters = 0
    elif len(designable) == 1:
        num_clusters = 1
    else:
        num_clusters, _ = _clusters_single_linkage(designable, tm_thresh=tm_cluster)

    # pclusters = (#designable clusters) / N_generated  (exact paper definition)
    pclusters = num_clusters / N

    # F1 (β=1)
    F1 = 0.0 if (pstructures + pclusters) == 0 else (2 * pstructures * pclusters) / (pstructures + pclusters)

    # Novelty (optional): among designable, keep those whose max TM to any reference ≤ 0.5,
    # then single-linkage cluster that novel subset with TM ≥ 0.6; novelty = (#novel clusters) / N
    novelty = None
    if ref_dir:
        ref_files = sorted(glob.glob(os.path.join(ref_dir, "*.pdb")))
        if not ref_files:
            raise RuntimeError("No PDBs found in reference directory for novelty.")
        novel = []
        for f in designable:
            # max TM to reference
            max_tm = 0.0
            for r in ref_files:
                tm = _tm_align(f, r)
                if tm > max_tm:
                    max_tm = tm
                # early exit if already > threshold
                if max_tm > tm_novel_max:
                    break
            if max_tm <= tm_novel_max:
                novel.append(f)
        if len(novel) == 0:
            novelty = 0.0
        elif len(novel) == 1:
            novelty = 1.0 / N
        else:
            n_clusters, _ = _clusters_single_linkage(novel, tm_thresh=tm_cluster)
            novelty = n_clusters / N

    return {
        "N": N,
        "designable": len(designable),
        "pstructures": pstructures,
        "num_designable_clusters": num_clusters,
        "pclusters": pclusters,
        "F1": F1,
        "novelty": novelty,
    }

# -------------------- per-length grouping (paper’s Fig. 6 style) --------------------
def per_length_metrics(gen_dir: str, sctm_map: dict, sctm_thresh=0.5, tm_cluster=0.6):
    """
    Groups by exact length (Cα count). For each length L:
      - pstructures_L = (#designable at length L)/N_L
      - pclusters_L   = (#clusters among designable at length L)/N_L
      - F1_L          = harmonic mean of pstructures_L and pclusters_L
    """
    # quick length reader (Cα lines per file)
    def ca_length(pdb_file: str) -> int:
        n=0
        with open(pdb_file) as f:
            for line in f:
                if line.startswith("ATOM") and line[12:16].strip() == "CA":
                    n+=1
        return n

    files = sorted(glob.glob(os.path.join(gen_dir, "*.pdb")))
    by_len = defaultdict(list)
    for f in files:
        by_len[ca_length(f)].append(f)

    rows = []
    for L, group in sorted(by_len.items()):
        N_L = len(group)
        designable_L = [f for f in group if sctm_map.get(os.path.basename(f), -1) >= sctm_thresh]
        pstruct_L = len(designable_L) / N_L
        if len(designable_L) <= 1:
            clusters_L = len(designable_L)
        else:
            clusters_L, _ = _clusters_single_linkage(designable_L, tm_thresh=tm_cluster)
        pclust_L = clusters_L / N_L
        F1_L = 0.0 if (pstruct_L + pclust_L) == 0 else (2 * pstruct_L * pclust_L) / (pstruct_L + pclust_L)
        rows.append({"length": L, "N": N_L, "pstructures": pstruct_L, "pclusters": pclust_L, "F1": F1_L})
    return rows

# Paths
gen_dir   = "/home/ubuntu/safegenie2/results/base/outputs/reproduce_2/pdbs"
sctm_csv  = "/home/ubuntu/safegenie2/results/base/outputs/reproduce_2/info.csv"  # or your JSON
ref_dir   = "/path/to/reference_pdbs"  # optional

# Top-level metrics (paper-faithful)
summary = compute_paper_metrics(
    gen_dir,
    sctm_csv=sctm_csv,
    sctm_thresh=0.5,
    tm_cluster=0.6,
    ref_dir=None,           # set to ref_dir for novelty
    tm_novel_max=0.5
)
print(summary)

# Per-length breakdown (paper Fig. 6)
sctm_map = load_sctm_csv(sctm_csv)  # or load_sctm_json(...)
per_len = per_length_metrics(gen_dir, sctm_map, sctm_thresh=0.5, tm_cluster=0.6)
for row in per_len:
    print(row)

