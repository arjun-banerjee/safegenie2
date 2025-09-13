import json
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Replace with your actual JSON file paths
json_files = [
    "training_losses_eta_single.json",
    "training_losses_eta_pair.json",
    "training_losses_eta_structure.json",
    "training_losses_eta_all.json",
]

# Custom titles for each run
run_titles = ["Single Feature", "Pair Feature", "Transformer", "All"]

# Collect data into a dataframe
all_data = []
for i, (f, title) in enumerate(zip(json_files, run_titles), 1):
    with open(f, "r") as infile:
        d = json.load(infile)
        for step, loss in enumerate(d["losses"]):
            all_data.append({"Step": step, "Loss": loss, "Run": title})

df = pd.DataFrame(all_data)


# Plot with seaborn
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x="Step", y="Loss", hue="Run", marker="o")

plt.title("Loss Curves Across Models")
plt.grid(True)

# Save instead of showing
plt.savefig("ablations.png", dpi=300, bbox_inches="tight")
plt.close()