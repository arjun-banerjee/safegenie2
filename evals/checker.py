import os
from biotite.structure import tm_score
from biotite.structure.io.pdb import PDBFile

def load_structure(pdb_path):
    """
    Load a PDB structure into an AtomArray.
    Only CA atoms are extracted for TM-score calculation.
    """
    pdb_file = PDBFile.read(pdb_path)
    array = pdb_file.get_structure(model=1)  # Get first model
    ca_atoms = array[array.atom_name == "CA"]
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
    
    # Compute TM-score
    score = tm_score(ref_structure, sample_structure)
    return score

def main():
    ref_path = "/Users/ethantam/desktop/1QLZ.pdb"
    sample_path = "/Users/ethantam/desktop/7X1U.pdb"
    score = compute_tm_score(ref_path, sample_path)
    print(f"TM-score between {os.path.basename(ref_path)} and {os.path.basename(sample_path)}: {score:.4f}")

if __name__ == "__main__":
    main()
