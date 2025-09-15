import os
from esm import pretrained, Alphabet, BatchConverter
from typing import List, Tuple, Optional, Dict, NamedTuple, Union, Callable
import itertools
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

import esm

torch.set_grad_enabled(False)


MAX_SEQUENCES = 10_000_000
MAX_LEN = 20000
DEVICE = "cuda"

esm2, esm2_alphabet = esm.pretrained.esm2_t6_8M_UR50D()
esm2 = esm2.eval().to(DEVICE)
esm2_batch_converter = esm2_alphabet.get_batch_converter()

input_fasta = "/home/ubuntu/uniref50.fasta"
output_csv = os.path.join(os.path.dirname(input_fasta), "uniprot50_perplexity.csv")


def compute_masked_perplexity_simplified(seq: str, lambda_tokens: int = 1):
    data = [("protein", seq)]
    _, _, tokens = esm2_batch_converter(data)
    tokens = tokens.to(DEVICE)
    
    seq_len = tokens.size(1)
    valid_mask = (tokens != esm2_alphabet.padding_idx) & (tokens != esm2_alphabet.cls_idx) & (tokens != esm2_alphabet.eos_idx)
    valid_indices = torch.where(valid_mask[0])[0]

    if len(valid_indices) < lambda_tokens:
        return None

    masked_indices = np.random.choice(valid_indices.cpu().numpy(), size=lambda_tokens, replace=False)
    masked_tokens = tokens.clone()
    masked_tokens[0, masked_indices] = esm2_alphabet.mask_idx

    with torch.no_grad():
        logits = esm2(masked_tokens)["logits"] 
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    target_tokens = tokens[0, masked_indices]  # [λ]
    pred_log_probs = log_probs[0, masked_indices].gather(1, target_tokens.unsqueeze(1)).squeeze(1)  # [λ]

    masked_perplexity = torch.exp(-pred_log_probs.mean()).item()
    return masked_perplexity


def compute_estimated_masked_perplexity(seq: str, lambda_t: int = 10, num_samples: int = 1000):
    """
    Monte-Carlo Estimate of Masked Perplexity
    
    lambda_t: divisor for fraction of tokens to mask away
    num_samples: permutations of masking to try and average
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


def compute_estimated_perplexity_by_cross_entropy(seq: str, lambda_t: int = 10, num_samples: int = 1000):
    """
    Monte-Carlo estimate of masked perplexity using cross-entropy loss.
    """
    lambda_tokens = max(1, len(seq) // lambda_t)
    data = [("protein", seq)]
    _, _, tokens = esm2_batch_converter(data)
    tokens = tokens.to(DEVICE)

    valid_mask = (tokens != esm2_alphabet.padding_idx) & \
                 (tokens != esm2_alphabet.cls_idx) & \
                 (tokens != esm2_alphabet.eos_idx)
    valid_indices = torch.where(valid_mask[0])[0]

    if len(valid_indices) < lambda_tokens:
        return None

    ce_losses = []

    for _ in range(num_samples):
        sampled_indices = np.random.choice(valid_indices.cpu().numpy(), size=lambda_tokens, replace=False)
        sampled_indices = torch.tensor(sampled_indices, device=DEVICE)

        masked_tokens = tokens.clone()
        masked_tokens[0, sampled_indices] = esm2_alphabet.mask_idx

        with torch.no_grad():
            logits = esm2(masked_tokens)["logits"]  # [1, seq_len, vocab_size]

        # Get logits and targets only at masked positions
        selected_logits = logits[0, sampled_indices, :]        # [λ, vocab_size]
        target_tokens = tokens[0, sampled_indices]             # [λ]

        # Compute cross-entropy loss
        loss = F.cross_entropy(selected_logits, target_tokens, reduction='mean')
        ce_losses.append(loss.item())

    avg_ce = np.mean(ce_losses)
    perplexity = np.exp(avg_ce)
    return perplexity
