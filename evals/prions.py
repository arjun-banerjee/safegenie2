import sys
import os
from biotite.structure import tm_score, superimpose_structural_homologs
from biotite.structure.io.pdb import PDBFile

# let's do a sanity check and check the structural similarity of prions

def load_structure(pdb_path):
    """
    Load a PDB structure into an AtomArray.
    Only CA atoms are extracted for TM-score calculation.
    """
    pdb_file = PDBFile.read(pdb_path)
    # print("pdb_file", pdb_file)
    array = pdb_file.get_structure(model=1)  # Get first model
    
    peptide_array = filter_peptide(array)
    # print("array", array)
    ca_atoms = peptide_array[peptide_array.atom_name == "CA"]
    return ca_atoms

def compute_tm_score(ref_path, sample_path):
    """
    Compute TM-score between a reference structure and a sample structure.
    Range of TM-score is [0, 1], where 1 indicates a perfect match.
    """
    # Load reference structure
    ref_structure = load_structure(ref_path)
    
    # Load sample structures
    sample_structure = load_structure(sample_path)
    
    # Superimpose sample onto reference
    superimposed, _, ref_indices, sub_indices = superimpose_structural_homologs(
        ref_structure, sample_structure, max_iterations=1
    )
    
    # Compute TM-score
    score = tm_score(ref_structure, superimposed, ref_indices, sub_indices)
    return score

def compute_all_tm_scores(ref_path, sample_paths):
    """
    ref path is the motif pdb path
    sample paths is a list of pdb paths to compare against the motif
    """
    scores = {}
    for sample_path in sample_paths:
        # print("currently processing:", sample_path)
        score = compute_tm_score(ref_path, sample_path)
        scores[os.path.basename(sample_path)] = score
    print("Average TM-score:", sum(scores.values()) / len(scores))
    return scores 

def load_pdbs_in_directory(directory):
    """
    Load all PDB files in a directory and return a list of their paths.
    """
    pdb_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.pdb')]
    return pdb_files


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

prions_pdbs_path = os.path.join(ROOT_DIR, "coord_datasets/prion_dataset/prion_pdbs")
pdbs = load_pdbs_in_directory(prions_pdbs_path)

reference_pdb = os.path.join(prions_pdbs_path, "1ag2.pdb")  # reference prion structure

reference_pdb = os.path.join(ROOT_DIR, "results/base/evals/half/motif=1e1j_processed/pdbs/1e1j_processed_0.pdb")


# # iterate over all pdbs and compute their tm scores against the reference
# scores = compute_all_tm_scores(reference_pdb, pdbs)
# print("Scores:", scores)


from Bio.PDB import PDBParser, PPBuilder

# Parse the PDB
parser = PDBParser(QUIET=True)
structure = parser.get_structure("protein", reference_pdb)

# Build polypeptides (chains)
ppb = PPBuilder()
for pp in ppb.build_peptides(structure):
    # This prints the sequence in FASTA format
    
    print(pp.get_sequence())