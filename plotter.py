import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys

def compute_ema(values, alpha=0.1):
    """Compute exponential moving average (EMA) smoothing."""
    ema = []
    current = values[0]
    for v in values:
        current = alpha * v + (1 - alpha) * current
        ema.append(current)
    return np.array(ema)

def plot_losses(json_file, output_file="loss_plot_eta.png"):
    # Load JSON file
    with open(json_file, "r") as f:
        data = json.load(f)

    losses = data.get("losses", [])
    if not losses:
        raise ValueError("No 'losses' found in JSON.")

    steps = np.arange(len(losses))
    smoothed = compute_ema(losses, alpha=0.05)  # adjust alpha for smoothing strength

    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6))

    # Plot raw loss
    sns.lineplot(x=steps, y=losses, marker="o", label="Loss", color="blue")

    # Plot smoothed loss (EMA)
    sns.lineplot(x=steps, y=smoothed, marker="o", color="red",
                 label="Smoothed Loss (EMA)", markersize=4)

    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Loss Plot for Eta = 10")
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_file)
    plt.close()
    print(f"Saved plot as {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_losses.py <input.json> [output.png]")
    else:
        json_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else "loss_plot.png"
        plot_losses(json_file, output_file)
