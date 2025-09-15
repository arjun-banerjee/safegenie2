import os
import seaborn as sns
import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

mpapa_vals_path = os.path.join(ROOT_DIR, "results/base/evals/half/motif=1e1j_processed/pdbs/fasta_pdbs/mpapa_vals.txt")

# load the mpapa_vals into a dictionary
mpapa_vals = {}
with open(mpapa_vals_path, "r") as f:
    for line in f:
        if ": " in line:
            key, val = line.strip().split(": ")
            mpapa_vals[key] = eval(val)  # convert string representation of list back to list
print("mpapa_vals:", mpapa_vals)
mpapa_data = mpapa_vals['sequences.txt']
print("mpapa_data:", mpapa_data)

# Optional: set a Seaborn style
sns.set(style="whitegrid")

# Create the histogram
plt.figure(figsize=(10, 6))
sns.histplot(mpapa_data, bins=20, kde=False, color="skyblue", edgecolor="black")

plt.title(f"Histogram of MPAPA Values (Base Model)")
plt.xlabel('MPAPA Value')
plt.ylabel('Frequency')
plt.ylim(top=65)

# add a red line for the mean value
mean_val = sum(mpapa_data) / len(mpapa_data)
plt.axvline(mean_val, color='red', linestyle='solid', linewidth=2)
# plt.text(mean_val + 0.1, plt.ylim()[1] * 0.9, f'Mean: {mean_val:.2f}', color='red')

# Save the figure
plt.savefig(os.path.join(ROOT_DIR, "results/base/evals/half/motif=1e1j_processed/pdbs/fasta_pdbs/mpapa_histogram.png"))
plt.close()
