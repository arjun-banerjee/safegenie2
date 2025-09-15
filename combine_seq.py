import os

# Root directory containing tester_1, tester_2, ...
root_dir = "results/base/evals/prion_eta_10_pdbs/outputs"
output_file = "results/base/evals/prion_eta_10_pdbs/outputs/eta10_prions_correct.txt"

with open(output_file, "w") as outfile:
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".fa"):
                filepath = os.path.join(subdir, file)
                with open(filepath, "r") as infile:
                    for line in infile:
                        line = line.strip()
                        # Skip header lines starting with ">"
                        if not line.startswith(">") and line != "":
                            outfile.write(line + "\n")

print(f"✅ All sequences have been saved to {output_file}")
