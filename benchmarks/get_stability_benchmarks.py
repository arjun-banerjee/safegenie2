import subprocess
import os
import shutil
import tempfile
import re
import torch
from transformers import AutoTokenizer, EsmForProteinFolding
import py3Dmol


# Load model and tokenizer (Hugging Face version)
tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
model.cuda().eval()



def seq_to_pdb(sequence: str, pdb_write_path: str) -> str:
    with torch.no_grad():
        output = model.infer(sequence)
        pdb_str = model.output_to_pdb(output)[0]
    # Save to file
    with open(pdb_write_path, "w") as f:
        f.write(pdb_str)
    return pdb_str


def visualize_pdb(pdb_path: str):
    with open(pdb_path) as f:
        pdb_data = f.read()

    view = py3Dmol.view(width=800, height=600)
    view.addModel(pdb_data, "pdb")
    view.setStyle({"cartoon": {"color": "spectrum"}})
    view.zoomTo()
    view.show()

def parse_stability_fxout(fxout_path):
    """Parse FoldX Stability .fxout file and return ΔG and full breakdown."""
    with open(fxout_path, 'r') as f:
        for line in f:
            if line.strip() == "":
                continue
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                try:
                    dG = float(parts[1])  # total ΔG of folding
                    breakdown = [float(x) if i > 0 else x for i, x in enumerate(parts)]
                    return {"dG": dG, "full": breakdown}
                except ValueError:
                    continue
    return None

def run_foldx_stability(pdb_path):
    # Resolve absolute paths
    pdb_path = os.path.abspath(pdb_path)
    pdb_filename = os.path.basename(pdb_path)
    pdb_basename = os.path.splitext(pdb_filename)[0]

    # Define FoldX paths (adjust if needed)
    foldx_exec = "/home/ubuntu/protolyze/benchmarks/foldx/foldx_20251231"
    rotabase_path = "/home/ubuntu/protolyze/benchmarks/foldx/rotabase.txt"

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




# sequence = "MNYRCVKNGRKCEMIESHERKKTMVIVNYYTLQAALHMLIKATHISRVRIDKGGEAMGMQACYPNNTNTQGGPELMLSCKVAFNTAIMPQDKGPRYLLGWWPADKGDTSANRRRWGQDA"
# PDB_HOME_DIR =  "/home/ubuntu/protolyze/benchmarks/foldx/pdbs/"
# pdb_name = "dummy_zinc_finger.pdb"
# output_pdb_path = PDB_HOME_DIR + "/" + pdb_name
# seq_to_pdb(sequence, output_pdb_path)
# result = run_foldx_stability(output_pdb_path) # repairs, then runs report
# print(result["energy"])
# print(result)

# visualize_pdb(output_pdb_path)
