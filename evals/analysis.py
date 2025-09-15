import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def count_pdb_rows(pdb_dir: str):
    """
    Counts the number of lines (rows) in each PDB file in the given directory.

    Args:
        pdb_dir (str): Path to directory containing PDB files.

    Returns:
        dict: Mapping from PDB filename to number of rows.
    """
    pdb_row_counts = {}
    count = 0
    for pdb_file in os.listdir(pdb_dir):
        if pdb_file.endswith(".pdb"):
            pdb_path = os.path.join(pdb_dir, pdb_file)
            try:
                with open(pdb_path, "r") as f:
                    num_rows = sum(1 for _ in f)
                pdb_row_counts[pdb_file] = num_rows
                count += 1
            except Exception as e:
                print(f"Error reading {pdb_file}: {e}")
                pdb_row_counts[pdb_file] = None
        if count == 20:
            break


    return pdb_row_counts


def count_helix_ranges_from_json(json_path):
    """
    Reads a JSON file with alpha-helix data and counts the length of each segment.

    Args:
        json_path (str): Path to JSON file.

    Returns:
        dict: Mapping PDB path -> segment lengths dict.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    results = {}
    
    count = 0
    
    res = []

    for pdb_path, pdb_data in data.items():
        filename = os.path.basename(pdb_path)
        count += 1
        # if count > 20:
        #     break
        #check if pdb_data is a dict and has "data" key
        if not isinstance(pdb_data, dict) or "data" not in pdb_data:
            print(f"Invalid data format for {pdb_path}, skipping.")
            continue
        # segment_counts = {}
        
        curr_sum = 0
        
        for seg_name, reported_length in pdb_data.get("data", {}).items():
            # Extract the range part after the colon
            range_part = seg_name.split(":")[1].strip()
            # Replace unicode en-dash with normal dash
            range_part = range_part.replace("\u2013", "-")
            try:
                start, end = map(int, range_part.split("-"))
                count = end - start + 1
                curr_sum += count
                # res.append(count)
            except Exception as e:
                print(f"Error parsing segment {seg_name} in {pdb_path}: {e}")
                count = None
            # segment_counts[seg_name] = count

        results[filename] = curr_sum

    return results


def plot_histogram(data, output_path, eta):
    # plt.figure(figsize=(10, 6))
    
    # sns.histplot(data, bins=range(0, 100), kde=False)
    
    # # Include the Greek symbol eta in the title
    # plt.title(f"Distribution of Alpha Helix Lengths (η = {eta})")
    
    # plt.xlabel("Alpha Helix Length")
    # plt.ylabel("Frequency")
    # plt.savefig(output_path)
    # plt.close()
    
    plt.figure(figsize=(10,6))
    sns.kdeplot(data, fill=True)
    plt.xlabel("Percentage of Atoms in Alpha Helices (%)")
    plt.ylabel("Density")
    plt.title(f"Distribution of Alpha-Helix Content Across PDBs (Base Model)")
    plt.xlim(0, 100)
    
    # Red vertical line at mean
    mean_val = np.mean(data)
    plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f"Mean = {mean_val:.2f}")
    
    plt.savefig(output_path)
    plt.close()
    
    
def plot_histogram_2(data, output_path):
    plt.figure(figsize=(10, 6))
    
    sns.histplot(data, bins=range(0, 40), kde=False)
    
    # Include the Greek symbol eta in the title
    plt.title(f"Percentage of Atoms in Alpha Helices (η = 10)")
    
    plt.xlabel("Percentage of Atoms in Alpha Helices (%)")
    plt.ylabel("Frequency")
    plt.savefig(output_path)
    plt.close()
    
    
# def plot_percentage_histogram(percentages, output_path):
#     """
#     Plots a histogram of alpha-helix percentages with count on the y-axis.
#     Works even if all values are 0.

#     Args:
#         percentages (list of float): Percentages of atoms in alpha helices per PDB.
#         output_path (str): Path to save the figure.
#     """
#     plt.figure(figsize=(10, 6))
    
#     # Ensure at least one bin for all zeros
#     if max(percentages) == 0:
#         bins = [0, 1]  # single bin for 0%
#     else:
#         bins = 10  # or choose a reasonable number of bins
    
#     sns.histplot(percentages, bins=bins, kde=False, color="skyblue", stat="count")
    
#     plt.xlabel("Percentage of Atoms in Alpha Helices (%)")
#     plt.ylabel("Count")
#     plt.title("Distribution of Alpha-Helix Content Across PDBs")
#     plt.xlim(0, max(percentages) if max(percentages) > 0 else 1)
    
#     plt.tight_layout()
#     plt.savefig(output_path)
#     plt.close()


def main():
    # pdb_dir = "/home/ubuntu/safegenie2/results/base/evals/arjun_checkpoint_005/pdbs/"
    
    # path = "/home/ubuntu/safegenie2/results/" + "alpha_helix_length_distribution_2.png"
    # plot_histogram_2([0] * 20, path)
    
    pdb_dir = "/home/ubuntu/safegenie2/results/base/outputs/unconditional_15/pdbs/"
    row_counts = count_pdb_rows(pdb_dir)
    print("row counts:", row_counts)
    
    json_path = pdb_dir + "alpha_helix_results_2.json"
    helix_counts = count_helix_ranges_from_json(json_path)
    print("helix counts:", helix_counts)
    
    print("row counts:", len(row_counts))
    print("helix counts:", len(helix_counts))
    print("percentage: ", len(helix_counts)/len(row_counts))
    
    data = []
    
    for key in row_counts:
        print("key:", key)
        num_rows = row_counts.get(key)
        print("num_rows:", num_rows)
        num_helix = helix_counts.get(key)
        print("num_helix:", num_helix)
        
        # find the number of atoms that are part of an alpha helix
        percentage = num_helix / num_rows
        
        data.append(percentage)

    print("len data:", len(data))
    
    data_100 = [x * 100 for x in data]
    
    plot_histogram(data_100, pdb_dir + "alpha_helix_length_distribution_2.png", eta=0.5)
    # plot_histogram(data_100, pdb_dir + "alpha_helix_length_distribution_2_fixed.png", eta=0.5)
    
    
    
    
    

if __name__ == "__main__":
    main()