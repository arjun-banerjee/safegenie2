import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# Path to your UniProt FASTA
uniprot_path = "/home/ubuntu/uniref50.fasta"
output_dir = os.path.dirname("biochemical_benchmarks/")  # save outputs to same dir

# Biochemical keys you're tracking
global_keys = ["length", "gravy", "instability_index", "isoelectric_point", "aromaticity", "charge_at_ph7"]
banned_symbols = {"X", "B", "Z", "U", "O"}  # Skip sequences with unknown amino acids

# Store results here
rows = []
ids = []

def get_basic_props(sequence: str):
    if any(c in banned_symbols for c in sequence):
        return None
    analyzed = ProteinAnalysis(str(sequence))
    return {
        "length": len(sequence),
        "gravy": analyzed.gravy(),
        "instability_index": analyzed.instability_index(),
        "isoelectric_point": analyzed.isoelectric_point(),
        "aromaticity": analyzed.aromaticity(),
        "charge_at_ph7": analyzed.charge_at_pH(7.0),
    }



# Process sequences

# QUERY 
def query():
    print("running query")
    MAX_SEQUENCES = 5_000_000
    with open(uniprot_path) as handle:
        for i, record in enumerate(SeqIO.parse(handle, "fasta")):
            if i >= MAX_SEQUENCES:
                break
            if i % 10 != 0:
                continue
            props = get_basic_props(record.seq)
            if props:
                rows.append(props)
                ids.append(record.id)
                
    # Create DataFrame
    df = pd.DataFrame(rows, columns=global_keys)
    df["sequence_id"] = ids
    csv_path = os.path.join(output_dir, "uniprot50_biochemical_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV to: {csv_path}")

    # Display summary statistics
    summary = df.describe()
    print("\nSummary Statistics:")
    print(summary)

    # Save summary as CSV too
    #summary_csv_path = os.path.join(output_dir, "uniprot50_biochemical_stats.csv")
    #summary.to_csv(summary_csv_path)

    # Plotting
    # sns.set(style="whitegrid")
    # for key in global_keys:
    #     plt.figure(figsize=(6, 4))
    #     sns.histplot(df[key], kde=True, bins=50)
    #     plt.title(f"{key} Distribution")
    #     plt.xlabel(key)
    #     plt.ylabel("Count")
    #     plot_path = os.path.join(output_dir, f"{key}_distribution.png")
    #     plt.tight_layout()
    #     plt.savefig(plot_path)
    #     plt.close()
    #     print(f"Saved plot: {plot_path}")


# query()