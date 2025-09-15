import os
from pathlib import Path

# Mapping of nonstandard 3-letter codes to canonical ones
RESIDUE_NORMALIZATION = {
    "SEC": "CYS",  # selenocysteine -> cysteine
    "MSE": "MET",  # selenomethionine -> methionine
    "HSD": "HIS", "HSE": "HIS", "HSP": "HIS",  # histidine variants
    "GLX": "GLU",  # ambiguous GLN/GLU -> GLU
    "ASX": "ASP",  # ambiguous ASN/ASP -> ASP
    "CSO": "CYS", "CSE": "CYS", "CSD": "CYS",  # oxidized cysteine variants
    "SEP": "SER", "TPO": "THR", "PTR": "TYR",  # phosphorylated residues
    # Add more if needed
}

def normalize_residue(res_name: str) -> str:
    """Collapse nonstandard residues to canonical equivalents."""
    res_name = res_name.upper().strip()
    return RESIDUE_NORMALIZATION.get(res_name, res_name)

def process_pdb_single_chain(input_path, output_dir):
    """
    Process a PDB file:
    - Keeps only one C-alpha atom per residue (highest occupancy)
    - Renumbers atom serial numbers and residues sequentially
    - Forces all residues into chain A
    - Collapses nonstandard residues into canonical equivalents
    - Skips proteins longer than 255 residues
    - Writes aligned PDB output with proper fixed-width formatting
    """
    name = Path(input_path).stem
    residues = {}

    with open(input_path, "r") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue

            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue

            res_seq = int(line[22:26].strip())
            occupancy = float(line[54:60].strip() or 0.0)

            # Keep only highest occupancy Cα per residue
            if res_seq not in residues or occupancy > residues[res_seq][0]:
                residues[res_seq] = (occupancy, line)

    # Sort residues by original residue sequence
    sorted_res = sorted(residues.items(), key=lambda x: x[0])
    total_length = len(sorted_res)

    if total_length > 255:
        print(f"⚠️ Skipped {name}, too long ({total_length} residues)")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{name}_processed.pdb"

    with open(output_path, "w") as out:
        out.write(f"REMARK 999 NAME   {name}\n")
        out.write(f"REMARK 999 PDB    {name}\n")
        out.write(f"REMARK 999 INPUT A   1 {total_length} A\n")
        out.write(f"REMARK 999 MINIMUM TOTAL LENGTH      {total_length}\n")
        out.write(f"REMARK 999 MAXIMUM TOTAL LENGTH      {total_length}\n")
        out.write("\n")

        atom_serial = 1
        for new_res_id, (res_seq, (occ, line_str)) in enumerate(sorted_res, start=1):
            record = line_str[0:6]
            atom_name = line_str[12:16]
            alt_loc = " "
            res_name = normalize_residue(line_str[17:20])  # normalize here
            chain_id = "A"
            x = float(line_str[30:38])
            y = float(line_str[38:46])
            z = float(line_str[46:54])
            occupancy = float(line_str[54:60])
            bfactor = float(line_str[60:66])
            element = line_str[76:78].strip().rjust(2)

            new_line = (
                f"{record}{atom_serial:5d} {atom_name:>4}{alt_loc}{res_name:>3} {chain_id}"
                f"{new_res_id:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}"
                f"{occupancy:6.2f}{bfactor:6.2f}          {element:>2}\n"
            )
            out.write(new_line)
            atom_serial += 1

        out.write("END\n")

    print(f"✅ Processed: {name} ({total_length} residues) → {output_path}")

def process_all_pdbs(input_folder, output_folder):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)

    pdb_files = list(input_folder.glob("*.pdb"))

    if not pdb_files:
        print("No PDB files found in input folder.")
        return

    for pdb_file in pdb_files:
        process_pdb_single_chain(pdb_file, output_folder)

if __name__ == "__main__":
    input_folder = ""
    output_folder = "data/PDB_files/processed_pdbs"
    process_all_pdbs(input_folder, output_folder)
