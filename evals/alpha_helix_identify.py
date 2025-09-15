from Bio.PDB import PDBParser, DSSP
import os

def has_alpha_helix(pdb_file):
    # in order to run this: 
    # conda install -c conda-forge dssp
    # also pip install biopython 
    # https://pdb-redo.eu/dssp/about
    """
    DSSP provides an elaborate description of the secondary structure elements in a protein structure, 
    including backbone hydrogen bonding and the topology of β-sheets. The most popular feature is the 
    per-residue assignment of secondary structure with a single character code:

    H = α-helix
    B = residue in isolated β-bridge
    E = extended strand, participates in β ladder
    G = 310-helix
    I = π-helix
    P = κ-helix (poly-proline II helix)
    T = hydrogen-bonded turn
    S = bend
    """

    """ Checks if PDB file contains alpha helices """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    model = structure[0]
    
    # run DSSP
    dssp = DSSP(model, pdb_file)
    
    # check for alpha helices (H) only
    for residue in dssp:
        secondary_structure = residue[2]  # secondary structure is at index 2
        if secondary_structure == 'H':  # alpha helix
            print("residue that is in the alpha helix:", residue)
            return True
    
    return False

# change pdb_file name:
pdb_file = "/home/ubuntu/safegenie2/results/base/outputs/reproduce_2/pdbs/114_3_relaxed_patched.pdb"
if has_alpha_helix(pdb_file):
    print("Alpha helix found!")
else:
    print("No alpha helix found.")