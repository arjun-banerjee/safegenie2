from Bio.PDB import PDBList
import os
import shutil

# List of PDB IDs
# from: https://pitgroup.org/apps/amyloid/amyloid_list
pdb_ids = [
    "1a15","1a64","1a6p","1ajy","1alv","1alw","1aoj","1arq","1arr","1auu",
    "1b28","1b49","1b5d","1b5e","1b6k","1b6m","1b6p","1baz","1bko","1bqp",
]

# Folder to save PDB files
output_folder_path = "/home/ubuntu/safegenie2/data/amyloid_pdbs"
os.makedirs(output_folder_path, exist_ok=True)

# Initialize PDB downloader
pdbl = PDBList()

for pdb_id in pdb_ids:
    # Download the .ent file
    ent_file = pdbl.retrieve_pdb_file(
        pdb_id,
        pdir=output_folder_path,
        file_format="pdb",
        overwrite=True
    )
    
    # Rename it to <PDB_ID>.pdb
    base_name = os.path.basename(ent_file)          # e.g., pdb1a15.ent
    new_name = os.path.join(output_folder_path, f"{pdb_id}.pdb")
    shutil.move(ent_file, new_name)

print(f"Downloaded {len(pdb_ids)} PDB files to {output_folder_path}")
