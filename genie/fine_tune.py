import torch
import torch.nn as nn
from torch.optim import Adam
from pytorch_lightning.core import LightningModule
from tqdm import tqdm
import os
import numpy as np

from genie.model.model import Denoiser
from genie.diffusion.schedule import get_betas
from genie.utils.affine_utils import T
from genie.utils.geo_utils import compute_frenet_frames
from genie.utils.feat_utils import (
    prepare_tensor_features,
    create_np_features_from_motif_pdb_spec,
    create_empty_np_features,
    convert_np_features_to_tensor,
    batchify_np_features
)
import matplotlib.pyplot as plt



class GenieFineTuner(LightningModule):
    """
    Fine-tuning class for Genie 2 model that implements concept erasure/adjustment
    with proper motif scaffolding support.
    """

    def __init__(
        self,
        pretrained_model_path,
        config,
        eta=1.0,
        learning_rate=1e-5,
        min_length=150,
        max_length=256,
        num_samples_per_step=4,
        motif_scaffolding_files=None
    ):
        """
        Args:
            motif_scaffolding_files: List of paths to motif scaffolding problem files
        """
        super(GenieFineTuner, self).__init__()
        self.config = config
        self.eta = eta
        self.learning_rate = learning_rate
        self.min_length = min_length
        self.max_length = max_length
        self.num_samples_per_step = num_samples_per_step
        self.motif_scaffolding_files = motif_scaffolding_files or []
        self._device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(self._device_str)
        
        # Load pretrained model
        self.model = self._load_pretrained_model(pretrained_model_path)
        
        # Create frozen reference model (original weights)
        self.frozen_model = self._load_pretrained_model(pretrained_model_path)
        self.frozen_model.eval()
        for param in self.frozen_model.parameters():
            param.requires_grad = False
            
        # Setup diffusion schedule
        self.setup_schedule()
        
        # Loss function
        self.loss_fn = nn.MSELoss()

    def _load_pretrained_model(self, model_path):
        """Load pretrained model from checkpoint"""
        from genie.diffusion.genie import Genie
        
        # Load configuration
        config = self.config
        
        # Check if it's a .ckpt file (PyTorch Lightning checkpoint)
        if model_path.endswith('.ckpt'):
            print(f'Loading checkpoint: {model_path}')
            return Genie.load_from_checkpoint(model_path, config=config)
        else:
            # Fallback for .pt files
            model = Denoiser(
                **self.config.model,
                n_timestep=self.config.diffusion['n_timestep'],
                max_n_res=self.config.io['max_n_res'],
                max_n_chain=self.config.io['max_n_chain']
            )
            
            # Load pretrained weights if available
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location='cpu')
                if 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            
            return model

    def setup_schedule(self):
        """Set up variance schedule and precompute terms"""
        # Move models to device first
        self.model = self.model.to(self._device)
        self.frozen_model = self.frozen_model.to(self._device)
        
        self.betas = get_betas(
            self.config.diffusion['n_timestep'],
            self.config.diffusion['schedule']
        ).to(self._device)

        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_cumprod_prev = torch.cat([
            torch.Tensor([1.]).to(self._device),
            self.alphas_cumprod[:-1]
        ])
        self.one_minus_alphas_cumprod = 1. - self.alphas_cumprod
        
        self.sqrt_betas = torch.sqrt(self.betas)
        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_alphas_cumprod_prev = torch.sqrt(self.alphas_cumprod_prev)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod_prev = torch.sqrt(1. - self.alphas_cumprod_prev)
        self.sqrt_recip_alphas_cumprod = 1. / self.sqrt_alphas_cumprod
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1)


    def generate_conditioned_features(self, motif_pdb_path):
        """
        Generate features for conditioned generation (with motif)
        """
        # Create conditioned features using Genie 2 utilities
        np_features = create_np_features_from_motif_pdb_spec(motif_pdb_path)
        
        self._last_generated_length = np_features['num_residues']
        
        # Convert to tensor and batch (same pattern as unconditioned)
        features = convert_np_features_to_tensor(
            batchify_np_features([np_features]), self._device
        )
        return features

    def generate_unconditioned_features(self, sequence_length):
        """
        Generate features for unconditioned generation (no motif)
        """
        # Create empty features using Genie 2 utilities
        np_features = create_empty_np_features([sequence_length])
        
        # Convert to tensor and batch
        features = convert_np_features_to_tensor(
            batchify_np_features([np_features]), self._device
        )
        return features

    def add_noise(self, features, timesteps):
        """
        Add noise to clean coordinates at given timesteps, respecting motif constraints
        """
        coords = features['atom_positions']
        batch_size = coords.shape[0]
        
        # Generate noise
        noise = torch.randn_like(coords)
        
        # Create mask for positions that CAN be noised (scaffold only)
        # valid positions AND not fixed (motifs)
        noise_mask = features['residue_mask'].unsqueeze(-1).float() * \
                    (~features['fixed_sequence_mask']).unsqueeze(-1).float()
        
        # Apply mask to noise - only noise scaffold positions
        masked_noise = noise * noise_mask
        
        # Get noise schedule coefficients
        timesteps = torch.clamp(timesteps, 0, len(self.sqrt_alphas_cumprod) - 1)
        sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[timesteps].view(batch_size, 1, 1)
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[timesteps].view(batch_size, 1, 1)
        
        # Add noise using reparameterization trick
        noisy_coords = sqrt_alpha_cumprod * coords + sqrt_one_minus_alpha_cumprod * masked_noise
        
        # Preserve original motif coordinates by copying them back
        motif_mask = features['fixed_sequence_mask'].unsqueeze(-1).float()
        noisy_coords = noisy_coords * (1 - motif_mask) + coords * motif_mask
        
        # Apply overall residue mask
        noisy_coords = noisy_coords * features['residue_mask'].unsqueeze(-1).float()
        
        # Compute frames from noisy coordinates
        rots = compute_frenet_frames(
            noisy_coords,
            features['chain_index'], 
            features['residue_mask']
        )
        ts = T(rots, noisy_coords)
        
        # Deep copy features
        noisy_features = {}
        for key, value in features.items():
            if torch.is_tensor(value):
                noisy_features[key] = value.clone()
            else:
                noisy_features[key] = value
                
        noisy_features['atom_positions'] = noisy_coords
        
        return noisy_features, ts, masked_noise

    def training_step(self):
        """
        Single fine-tuning step with shared x_t for cond/uncond
        and scaffold-only masked loss.
        """
        # 1) Build ONE conditioned clean sample
        cond_clean = self.generate_conditioned_features(
            np.random.choice(self.motif_scaffolding_files)
        )

        def clone_with_coords(feat, coords):
            out = {}
            for k, v in feat.items():
                if torch.is_tensor(v):
                    out[k] = v.clone()
                else:
                    out[k] = v
            out['atom_positions'] = coords.clone()
            return out

        total_loss = 0.0

        for _ in range(self.num_samples_per_step):
            # 2) Sample ONE timestep and ONE noise; create ONE shared x_t
            t = torch.randint(0, self.config.diffusion['n_timestep'], (1,),
                            device=self._device)
            coords = cond_clean['atom_positions']  # clean coords
            eps = torch.randn_like(coords)

            sqrt_ac = self.sqrt_alphas_cumprod[t].view(1, 1, 1)
            sqrt_om = self.sqrt_one_minus_alphas_cumprod[t].view(1, 1, 1)
            x_t = sqrt_ac * coords + sqrt_om * eps  # shared noisy sample

            # 3) Create BOTH feature views that reference the SAME x_t
            cond_noisy = clone_with_coords(cond_clean, x_t)
            uncond_noisy = clone_with_coords(cond_clean, x_t)
            # "Unconditioned" = zero out motif mask
            uncond_noisy['fixed_sequence_mask'] = torch.zeros_like(
                uncond_noisy['fixed_sequence_mask']
            )

            # 4) Build frames once from x_t and reuse
            rots = compute_frenet_frames(
                x_t, cond_noisy['chain_index'], cond_noisy['residue_mask']
            )
            ts = T(rots, x_t)

            # 5) Teacher predictions on SAME x_t (cond vs uncond)
            with torch.no_grad():
                frozen_cond_out = self.frozen_model.model(ts, t, cond_noisy)
                frozen_uncond_out = self.frozen_model.model(ts, t, uncond_noisy)
                frozen_cond_noise = frozen_cond_out['z']
                frozen_uncond_noise = frozen_uncond_out['z']

            # Target = ε_u − η (ε_c − ε_u)  = (1+η) ε_u − η ε_c
            target_noise = frozen_uncond_noise - self.eta * (
                frozen_cond_noise - frozen_uncond_noise
            )

            scale = frozen_uncond_noise.norm(dim=-1, keepdim=True) / (target_noise.norm(dim=-1, keepdim=True) + 1e-8)
            target_noise = target_noise * scale

            # 6) Student prediction (conditioned view)
            pred = self.model.model(ts, t, cond_noisy)['z']

            # 7) Scaffold-only masked L2 loss (ignore fixed motifs)
            loss = self.loss_fn(pred, target_noise)

            total_loss += loss

        avg_loss = total_loss / self.num_samples_per_step
        return avg_loss

    def configure_optimizers(self):
        """Configure optimizer for fine-tuning"""
        return Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4
        )

    def fine_tune(self, num_steps=1000, accum_steps=4, save_path=None):
        """
        Run fine-tuning with gradient accumulation and random scaffold batching.

        Args:
            num_steps: total number of optimizer updates
            accum_steps: number of scaffolds to accumulate before each update
        """
        print(f"Starting fine-tuning for {num_steps} steps...")
        print(f"Learning rate: {self.learning_rate}, Eta: {self.eta}")
        print(f"Length range: {self.min_length}–{self.max_length}")
        
        self.train()
        optimizer = self.configure_optimizers()
        self.losses = []

        step = 0
        while step < num_steps:
            optimizer.zero_grad()
            batch_loss = 0.0

            for i in range(accum_steps):
                # === Sample a random scaffold length ===
                seq_len = np.random.randint(self.min_length, self.max_length + 1)

                # === Generate conditioned features from a random motif ===
                motif_file = np.random.choice(self.motif_scaffolding_files)
                cond_clean = self.generate_conditioned_features(motif_file)

                # (Optionally adjust cond_clean to seq_len here if you want exact length match)
                # For now, just proceed with cond_clean's length

                # === Training step (loss for one scaffold) ===
                loss = self.training_step()
                loss = loss / accum_steps   # normalize
                loss.backward()
                batch_loss += loss.item()

            # === Optimizer update ===
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

            step += 1
            avg_loss = batch_loss / accum_steps
            self.losses.append(avg_loss)
            print(f"Step {step}/{num_steps} - Average Loss: {avg_loss:.6f}")

    def save_model(self, save_path):
        """Save the fine-tuned model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'eta': self.eta
        }, save_path)
        print(f"Model saved to: {save_path}")


def main():
    """Example usage of the fine-tuner"""
    from genie.config import Config
    
    # Load configuration
    config = Config()
    
    # Define motif scaffolding problem files
    motif_scaffolding_files = [
        "../data/design25/1bcf.pdb",
        # Add more scaffolding problem files here
    ]
    
    # Initialize fine-tuner
    fine_tuner = GenieFineTuner(
        pretrained_model_path="../results/base/checkpoints/epoch=40.ckpt",
        config=config,
        eta=0.9,  # Adjust this to control concept adjustment strength
        learning_rate=1e-5,
        min_length=3,
        max_length=5,
        num_samples_per_step=6,
        motif_scaffolding_files=motif_scaffolding_files
    )
    
    n_s = 30
    # Run fine-tuning
    fine_tuner.fine_tune(
        num_steps = n_s,
        save_path="fine_tuned_models"
    )
    
    # Save final model
    plt.figure(figsize=(8,5))
    plt.scatter(range(1, n_s+1), fine_tuner.losses, label="Training Loss", color="blue")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Fine-tuning Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_curve.png")   # save as file


if __name__ == "__main__":
    main()