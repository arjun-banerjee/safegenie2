import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Read the CSV file
csv_path = os.path.join(script_dir, 'data', 'base_output.csv')
df = pd.read_csv(csv_path)

# Filter data based on conditions
scRMSD_based = df[(df['scRMSD'] <= 2) & (df['pLDDT'] >= 70)]
scTM_based = df[(df['scTM'] >= 0.5) & (df['pLDDT'] >= 70)]

# Calculate percentages
total_domains = len(df)
scRMSD_percentage = (len(scRMSD_based) / total_domains) * 100
scTM_percentage = (len(scTM_based) / total_domains) * 100

# Create the bar chart
fig, ax = plt.subplots(figsize=(8, 6))

# Define colors for the bars (blue and yellow)
colors = ['#2E86AB', '#FFC107']  # Blue and yellow
bars = ax.bar(['scRMSD-based\n(scRMSD ≤ 2 & pLDDT ≥ 70)', 
               'scTM-based\n(scTM ≥ 0.5 & pLDDT ≥ 70)'],
              [scRMSD_percentage, scTM_percentage],
              color=colors,
              edgecolor='black',
              linewidth=1.5)

# Add value labels on top of bars
for i, (bar, percentage) in enumerate(zip(bars, [scRMSD_percentage, scTM_percentage])):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{percentage:.1f}%',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

# Customize the plot
ax.set_ylabel('Percentage of Designable Structures (%)', fontsize=12, fontweight='bold')
ax.set_title('Percentage of Designable Structures (Base Model)', fontsize=14, fontweight='bold', pad=20)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

# Adjust y-axis to show all bars clearly (0-100%)
ax.set_ylim(0, max(scRMSD_percentage, scTM_percentage) * 1.15)

plt.tight_layout()
output_path = os.path.join(script_dir, 'bar_chart.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()  # Close the figure to free up memory

# Print summary statistics
scRMSD_count = len(scRMSD_based)
scTM_count = len(scTM_based)
print(f"Total domains: {total_domains}")
print(f"scRMSD-based (scRMSD ≤ 2 & pLDDT ≥ 70): {scRMSD_count} ({scRMSD_percentage:.1f}%)")
print(f"scTM-based (scTM ≥ 0.5 & pLDDT ≥ 70): {scTM_count} ({scTM_percentage:.1f}%)")

