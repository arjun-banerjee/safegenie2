from Bio.PDB import PDBParser, DSSP
from biotite.structure.io.pdb import PDBFile
import sys
import os

def load_structure(pdb_path):
    """
    Load a PDB structure into an AtomArray.
    Only CA atoms are extracted for TM-score calculation.
    """
    pdb_file = PDBFile.read(pdb_path)
    # print("pdb_file", pdb_file)
    array = pdb_file.get_structure(model=1)  # Get first model
    # print("array", array)
    return array
    # ca_atoms = array[array.atom_name == "CA"]
    # return ca_atoms

def load_pdbs_in_directory(directory):
    """
    Load all PDB files in a directory and return a list of their paths.
    """
    pdb_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.pdb')]
    return pdb_files


def compute_beta_content(pdb_path):
    """
    Compute the fraction of residues in beta-sheet conformation using DSSP.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("struct", pdb_path)
    model = structure[0]  # first model
    dssp = DSSP(model, pdb_path)

    total_residues = len(dssp)
    beta_count = sum(1 for k in dssp.keys() if dssp[k][2] == "E")  # E = extended strand

    frac_beta = beta_count / total_residues if total_residues > 0 else 0
    return frac_beta

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
amyloid_pdbs_path = os.path.join(ROOT_DIR, "data/amyloid_pdbs")
pdbs = load_pdbs_in_directory(amyloid_pdbs_path)

for i in range(len(pdbs)):
    output = compute_beta_content(pdbs[i])
    print("Fraction of beta-sheet content in {}: {:.2f}".format(os.path.basename(pdbs[i]), output))
