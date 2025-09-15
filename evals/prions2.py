import os
from Bio import SeqIO
from Bio.PDB import PDBParser, PPBuilder
from Bio.SeqUtils import seq1
import subprocess

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# -------------------
# Step 1: Convert PDB -> FASTA
# -------------------
# def pdb_to_fasta(pdb_path, fasta_path):
#     parser = PDBParser(QUIET=True)
#     structure = parser.get_structure("protein", pdb_path)
    
#     ppb = PPBuilder()
#     with open(fasta_path, "w") as fasta_file:
#         print("Processing fasta:", fasta_path)
#         yo = ppb.build_peptides(structure)
#         print("yo:", yo)
#         for i, pp in enumerate(ppb.build_peptides(structure)):
#             print("hhiiii")
#             seq = pp.get_sequence()
#             print("seq:", seq)
#             fasta_file.write(f">chain_{i}_{os.path.basename(pdb_path)}\n")
#             fasta_file.write(str(seq) + "\n")

def pdb_to_fasta(pdb_path, fasta_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)

    with open(fasta_path, "w") as fasta_file:
        for model in structure:
            for chain in model:
                residues = [res for res in chain if res.id[0] == " "]  # exclude heteroatoms
                seq = "".join(seq1(res.resname) for res in residues)
                fasta_file.write(f">{chain.id}_{os.path.basename(pdb_path)}\n")
                fasta_file.write(seq + "\n")

# -------------------
# Step 2: Batch convert all PDBs in directory
# -------------------
def convert_all_pdbs(pdb_dir):
    fasta_dir = os.path.join(pdb_dir, "fasta_pdbs")
    os.makedirs(fasta_dir, exist_ok=True)
    
    fasta_files = []
    
    for filename in os.listdir(pdb_dir):
        if filename.endswith(".pdb"):
            pdb_path = os.path.join(pdb_dir, filename)
            print("Converting:", pdb_path)
            fasta_path = os.path.join(fasta_dir, filename.replace(".pdb", ".fasta"))
            pdb_to_fasta(pdb_path, fasta_path)
            fasta_files.append(fasta_path)
    
    return fasta_files

# -------------------
# Step 3: Run function on each FASTA
# -------------------
def my_function_on_fasta(fasta_file):
    """Example function: just prints ID and length of each sequence."""
    for record in SeqIO.parse(fasta_file, "fasta"):
        print(f"File: {fasta_file}, ID: {record.id}, Length: {len(record.seq)}")
        
        
def run_mpapa_on_dir(fasta_dir):
    results = {}
    for file in os.listdir(fasta_dir):
        if file.endswith(".txt"):
            fasta_path = os.path.join(fasta_dir, file)
            # Run `python mpapa.py <fasta_path>`
            output = subprocess.check_output(["python", "mpapa.py", fasta_path], text=True)
            
            print("output:", output)
            # Assume mpapa.py prints a list of numbers like: [1, 2, 3]
            numbers = eval(output.strip())  # Or use json.loads if mpapa.py prints JSON
            results[file] = numbers
    return results

# -------------------
# Example usage
# -------------------
if __name__ == "__main__":
    pdb_dir = os.path.join(ROOT_DIR, "results/base/evals/half/motif=1e1j_processed_eta_10/pdbs/")
    fasta_files = convert_all_pdbs(pdb_dir)
    # print("fasta_files:", fasta_files)
    print(f"Converted {len(fasta_files)} PDB files to FASTA format.")
    
    # mpapa_vals = run_mpapa_on_dir(fasta_dir=os.path.join(pdb_dir, "fasta_pdbs"))
    mpapa_vals = run_mpapa_on_dir(fasta_dir=os.path.join(pdb_dir))
    print("mpapa_vals:", mpapa_vals)
    
    # save this dictionary to a text file in the fasta directory
    with open(os.path.join(pdb_dir, "fasta_pdbs", "mpapa_vals.txt"), "w") as f:
        for k, v in mpapa_vals.items():
            f.write(f"{k}: {v}\n")
    
    
    # # Call function on each fasta
    # for fasta in fasta_files:
    #     my_function_on_fasta(fasta)
