# %%
import os
import torch
import pandas as pd
from esm import pretrained, Alphabet, BatchConverter

# %%
from typing import List, Tuple, Optional, Dict, NamedTuple, Union, Callable
import itertools
import os
import string
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.distance import squareform, pdist, cdist
import matplotlib.pyplot as plt
import matplotlib as mpl
from Bio import SeqIO
from tqdm import tqdm
import pandas as pd
import seaborn as sns

import esm

torch.set_grad_enabled(False)

# %%
esm2, esm2_alphabet = esm.pretrained.esm2_t6_8M_UR50D()
esm2 = esm2.eval().to("cuda")
esm2_batch_converter = esm2_alphabet.get_batch_converter()

# %%
input_fasta = "/home/ubuntu/uniref50.fasta"
output_csv = os.path.join(os.path.dirname(input_fasta), "uniprot50_perplexity.csv")

# %%
MAX_SEQUENCES = 500_000
MAX_LEN = 4000 # set by resources
DEVICE = "cuda"

# %%
def compute_masked_perplexity(seq: str, lambda_tokens: int = 1):
    data = [("protein", seq)]
    _, _, tokens = esm2_batch_converter(data)
    tokens = tokens.to(DEVICE)
    
    seq_len = tokens.size(1)
    valid_mask = (tokens != esm2_alphabet.padding_idx) & (tokens != esm2_alphabet.cls_idx) & (tokens != esm2_alphabet.eos_idx)
    valid_indices = torch.where(valid_mask[0])[0]

    if len(valid_indices) < lambda_tokens:
        return None  # too short

    # Sample λ random positions to mask
    masked_indices = np.random.choice(valid_indices.cpu().numpy(), size=lambda_tokens, replace=False)
    masked_tokens = tokens.clone()
    masked_tokens[0, masked_indices] = esm2_alphabet.mask_idx

    with torch.no_grad():
        logits = esm2(masked_tokens)["logits"]  # [1, seq_len, vocab_size]
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    # Gather true labels
    target_tokens = tokens[0, masked_indices]  # [λ]
    pred_log_probs = log_probs[0, masked_indices].gather(1, target_tokens.unsqueeze(1)).squeeze(1)  # [λ]

    # Compute perplexity
    masked_perplexity = torch.exp(-pred_log_probs.mean()).item()
    return masked_perplexity

# %%
def estimate_masked_perplexity(seq: str, lambda_t: int = 10, num_samples: int = 100):
    """
    Monte-Carlo Estimate of Masked Perplexity
    """
    lambda_tokens = max(1, len(seq) // lambda_t)
    data = [("protein", seq)]
    _, _, tokens = esm2_batch_converter(data)
    tokens = tokens.to(DEVICE)

    seq_len = tokens.size(1)
    valid_mask = (tokens != esm2_alphabet.padding_idx) & (tokens != esm2_alphabet.cls_idx) & (tokens != esm2_alphabet.eos_idx)
    valid_indices = torch.where(valid_mask[0])[0]

    if len(valid_indices) < lambda_tokens:
        return None

    log_prob_samples = []

    for _ in range(num_samples):
        sampled_indices = np.random.choice(valid_indices.cpu().numpy(), size=lambda_tokens, replace=False)
        masked_tokens = tokens.clone()
        masked_tokens[0, sampled_indices] = esm2_alphabet.mask_idx

        with torch.no_grad():
            logits = esm2(masked_tokens)["logits"]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        target_tokens = tokens[0, sampled_indices]
        pred_log_probs = log_probs[0, sampled_indices].gather(1, target_tokens.unsqueeze(1)).squeeze(1)
        log_prob_samples.append(pred_log_probs.mean().item())

    avg_log_prob = np.mean(log_prob_samples)
    perplexity = np.exp(-avg_log_prob)
    return perplexity


# %%
sequences = []
perplexities = []
ids = []


# %%
id_set = set(ids)

# %%
import csv

# Make sure output directory exists
os.makedirs("biochemical_benchmarks", exist_ok=True)
output_csv_path = "biochemical_benchmarks/perplexity_sample_1.csv"

# Only write header if the file doesn't exist
write_header = not os.path.exists(output_csv_path)

with open(input_fasta) as handle, open(output_csv_path, "a", newline="") as f_out:
    writer = csv.DictWriter(f_out, fieldnames=["unprot_id", "perplexity", "sequence"])
    if write_header:
        writer.writeheader()

    for i, record in enumerate(SeqIO.parse(handle, "fasta")):
        if i >= MAX_SEQUENCES:
            break
        seq = str(record.seq)
        if "X" in seq or len(seq) > MAX_LEN:
            continue
        if record.id in id_set:
            print("skipping")
            continue
        if i % 4 != 0:
            continue
        try:
            ppl = estimate_masked_perplexity(seq, lambda_t=10, num_samples=1000)
            print(f"[{i}] {record.id}: Perplexity = {ppl:.2f}")
            writer.writerow({"unprot_id": record.id, "perplexity": ppl, "sequence": seq})
            f_out.flush()  # Immediately write to disk
            id_set.add(record.id)
        except Exception as e:
            print(f"Error on sequence {record.id}: {e}")



# %%
plt.hist(perplexities, bins=50)
plt.title("Monte-Carlo Masked Perplexities")
plt.xlabel("Mean Perplexity Score")
plt.ylabel("Frequency")


# %%
df = pd.DataFrame(
    {
    "sequence" : sequences,
    "perplexity" : perplexities,
    "unprot_id" : ids
    }
)

#df.to_csv("biochemical_benchmarks/perplexity_sample_1.csv")
df.describe().to_csv("biochemical_benchmarks/perplexity_sample_1_simple_stats.csv")
df.describe()

# %%
sample_seq = "MNYRCVKNGRKCEMIESHERKKTMVIVNYYTLQAALHMLIKATHISRVRIDKGGEAMGMQACYPNNTNTQGGPELMLSCKVAFNTAIMPQDKGPRYLLGWWPADKGDTSANRRRWGQDA"
new_sample_score = estimate_masked_perplexity(sample_seq, lambda_t=10, num_samples=100)

sns.set(style="whitegrid")
plt.figure(figsize=(6, 4))
sns.histplot(perplexities, bins=50, kde=True)
plt.axvline(new_sample_score, color='red', linestyle='--', linewidth=2, label='Steered Zinc Finger')
plt.title("Monte-Carlo Masked Perplexities")
plt.xlabel("Mean Perplexity Score")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.show()


# %%



