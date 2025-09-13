import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from pytorch_lightning.core import LightningModule
from tqdm import tqdm
import os
import numpy as np
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import json

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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StabilizedGenieFineTuner(LightningModule):
    """
    Stabilized fine-tuning class for Genie 2 model with improved numerical stability,
    proper loss masking, and learning rate scheduling.
    """

    def __init__(
        self,
        pretrained_model_path,
        config,
        eta=1.0,
        learning_rate=1e-6,  # Reduced default LR
        min_length=150,
        max_length=256,
        num_samples_per_step=4,
        motif_scaffolding_files=None,
        warmup_steps=50,
        max_grad_norm=0.5,  # Reduced gradient clipping
        eta_schedule="constant",  # "constant", "linear_decay", "cosine_decay"
        loss_type="masked_mse",  # "masked_mse", "weighted_mse"
        validation_interval=10
    ):
        super(StabilizedGenieFineTuner, self).__init__()
        self.config = config
        self.eta = eta
        self.initial_eta = eta
        self.learning_rate = learning_rate
        self.min_length = min_length
        self.max_length = max_length
        self.num_samples_per_step = num_samples_per_step
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.eta_schedule = eta_schedule
        self.loss_type = loss_type
        self.validation_interval = validation_interval
        
        # Validate motif files
        self.motif_scaffolding_files = self._validate_motif_files(motif_scaffolding_files or [])
        
        self._device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(self._device_str)
        
        logger.info(f"Using device: {self._device_str}")
        logger.info(f"Found {len(self.motif_scaffolding_files)} motif files")
        logger.info(f"Initial eta: {self.eta}, LR: {self.learning_rate}")
        
        # Load models
        self.model = self._load_pretrained_model(pretrained_model_path)
        self.frozen_model = self._load_pretrained_model(pretrained_model_path)
        self.frozen_model.eval()
        for param in self.frozen_model.parameters():
            param.requires_grad = False
            
        # Setup diffusion schedule
        self.setup_schedule()
        
        # Loss function - Fixed: Use MSE loss instead of CosineSimilarity
        self.loss_fn = nn.MSELoss(reduction='none')
        
        # Training metrics
        self.losses = []
        self.grad_norms = []
        self.eta_values = []
        self.step_count = 0
        
        # EMA for loss smoothing
        self.loss_ema = None
        self.ema_decay = 0.9

    def _validate_motif_files(self, motif_files):
        """Validate motif files exist"""
        valid_files = []
        for file_path in motif_files:
            if os.path.exists(file_path) and file_path.endswith('.pdb'):
                valid_files.append(file_path)
            else:
                logger.warning(f"Motif file not found: {file_path}")
        
        if not valid_files:
            raise ValueError("No valid motif scaffolding files provided")
        return valid_files

    def _load_pretrained_model(self, model_path):
        """Load pretrained model with error handling"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        try:
            if model_path.endswith('.ckpt'):
                from genie.diffusion.genie import Genie
                return Genie.load_from_checkpoint(model_path, config=self.config)
            else:
                model = Denoiser(
                    **self.config.model,
                    n_timestep=self.config.diffusion['n_timestep'],
                    max_n_res=self.config.io['max_n_res'],
                    max_n_chain=self.config.io['max_n_chain']
                )
                checkpoint = torch.load(model_path, map_location='cpu')
                if 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                elif 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def setup_schedule(self):
        """Setup diffusion schedule"""
        self.model = self.model.to(self._device)
        self.frozen_model = self.frozen_model.to(self._device)
        
        self.betas = get_betas(
            self.config.diffusion['n_timestep'],
            self.config.diffusion['schedule']
        ).to(self._device)

        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        
        # Precompute terms
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

    def update_eta(self, step, total_steps):
        """Update eta according to schedule"""
        if self.eta_schedule == "linear_decay":
            self.eta = self.initial_eta * (1 - step / total_steps)
        elif self.eta_schedule == "cosine_decay":
            self.eta = self.initial_eta * 0.5 * (1 + np.cos(np.pi * step / total_steps))
        # else: constant eta
        
        self.eta_values.append(self.eta)

    def generate_conditioned_features(self, motif_pdb_path):
        """Generate conditioned features"""
        try:
            np_features = create_np_features_from_motif_pdb_spec(motif_pdb_path)
            features = convert_np_features_to_tensor(
                batchify_np_features([np_features]), self._device
            )
            return features
        except Exception as e:
            logger.error(f"Error generating features from {motif_pdb_path}: {e}")
            raise

    def compute_stable_target(self, frozen_cond_noise, frozen_uncond_noise, eta):
        """
        Compute stabilized concept adjustment target with gradient scaling
        """
        # Original target
        raw_target = frozen_uncond_noise - eta * (frozen_cond_noise - frozen_uncond_noise)
        
        # Option 1: Normalize to prevent explosion
        frozen_uncond_norm = frozen_uncond_noise.norm(dim=-1, keepdim=True)
        target_norm = raw_target.norm(dim=-1, keepdim=True)
        
        # Scale target to have similar magnitude as frozen_uncond
        scale_factor = frozen_uncond_norm / (target_norm + 1e-8)
        # Clamp scaling to prevent extreme values
        scale_factor = torch.clamp(scale_factor, 0.1, 2.0)
        
        stable_target = raw_target * scale_factor
        
        return stable_target

    def training_step(self, step, total_steps):
        """
        Stabilized training step with better loss computation
        """
        # Update eta schedule
        self.update_eta(step, total_steps)
        
        # Sample motif file
        motif_file = np.random.choice(self.motif_scaffolding_files)
        cond_clean = self.generate_conditioned_features(motif_file)

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
        valid_samples = 0

        for _ in range(self.num_samples_per_step):
            try:
                # Sample timestep and create shared noisy sample
                t = torch.randint(0, self.config.diffusion['n_timestep'], (1,), device=self._device)
                coords = cond_clean['atom_positions']
                eps = torch.randn_like(coords)

                # Add noise
                sqrt_ac = self.sqrt_alphas_cumprod[t].view(1, 1, 1)
                sqrt_om = self.sqrt_one_minus_alphas_cumprod[t].view(1, 1, 1)
                x_t = sqrt_ac * coords + sqrt_om * eps

                # Create conditioned and unconditioned views
                cond_noisy = clone_with_coords(cond_clean, x_t)
                uncond_noisy = clone_with_coords(cond_clean, x_t)
                uncond_noisy['fixed_sequence_mask'] = torch.zeros_like(
                    uncond_noisy['fixed_sequence_mask']
                )

                # Compute frames
                rots = compute_frenet_frames(
                    x_t, cond_noisy['chain_index'], cond_noisy['residue_mask']
                )
                ts = T(rots, x_t)

                # Teacher predictions
                with torch.no_grad():
                    frozen_cond_out = self.frozen_model.model(ts, t, cond_noisy)
                    frozen_uncond_out = self.frozen_model.model(ts, t, uncond_noisy)
                    frozen_cond_noise = frozen_cond_out['z']
                    frozen_uncond_noise = frozen_uncond_out['z']

                # Compute stable target
                target_noise = self.compute_stable_target(
                    frozen_cond_noise, frozen_uncond_noise, self.eta
                )

                # Student prediction
                pred = self.model.model(ts, t, cond_noisy)['z']

                # Compute loss with proper masking
                loss = self.compute_masked_loss(pred, target_noise, cond_noisy)
                
                # Check for NaN/Inf
                if torch.isfinite(loss):
                    total_loss += loss
                    valid_samples += 1
                else:
                    logger.warning(f"Invalid loss detected at step {step}, skipping sample")

            except Exception as e:
                logger.warning(f"Error in training step {step}: {e}")
                continue

        if valid_samples == 0:
            logger.error(f"No valid samples in step {step}")
            return torch.tensor(0.0, device=self._device, requires_grad=True)

        avg_loss = total_loss / valid_samples
        return avg_loss
        
    def save_losses_json(self, filename="training_losses.json"):
        """Save training losses, EMA, grads, and eta values to JSON"""
        data = {
            "losses": self.losses,
            "grad_norms": self.grad_norms,
            "eta_values": self.eta_values,
        }
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Training losses saved to {filename}")

    def compute_masked_loss(self, pred, target, features):
        """
        Compute loss ONLY on scaffold regions (excluding fixed motifs)
        
        Mask interpretation:
        - fixed_sequence_mask = 1 → motif (don't train on)  
        - fixed_sequence_mask = 0 → scaffold (train on)
        """
        # Create scaffold mask - only compute loss on non-motif regions
        # scaffold_mask = 1 where we want to compute loss (scaffold regions)
        scaffold_mask = features['fixed_sequence_mask'].float()  # [B, N] - Fixed: inverted mask
        residue_mask = features['residue_mask'].float()  # [B, N]
        
        # Combined mask - scaffold AND valid residues
        # final_mask = 1 where we compute loss
        final_mask = scaffold_mask * residue_mask  # [B, N]
        
        # Expand mask to match prediction dimensions
        if pred.dim() == 3:  # [B, N, C]
            mask = final_mask.unsqueeze(-1)  # [B, N, 1]
        elif pred.dim() == 4:  # [B, N, M, C] 
            mask = final_mask.unsqueeze(-1).unsqueeze(-1)  # [B, N, 1, 1]
        else:
            mask = final_mask
        
        # Compute per-element loss using MSE
        element_loss = self.loss_fn(pred, target)  # Same shape as pred/target
        
        # Apply mask - only keep loss from scaffold regions
        masked_loss = element_loss * mask
        
        # Compute mean over valid scaffold elements only
        valid_elements = mask.sum()
        if valid_elements > 0:
            loss = masked_loss.sum() / valid_elements
        else:
            # No scaffold regions to train on - this shouldn't happen
            logger.warning("No scaffold regions found for loss computation!")
            loss = torch.tensor(0.0, device=pred.device, requires_grad=True)
            
        return loss

    def configure_optimizers(self):
        """Configure optimizer with learning rate scheduling"""
        optimizer = Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-6,  # Reduced weight decay
            eps=1e-8
        )
        return optimizer

    def fine_tune(self, num_steps=1000, accum_steps=32, save_path=None):
        """
        Run stabilized fine-tuning with monitoring
        """
        logger.info(f"Starting stabilized fine-tuning for {num_steps} steps")
        logger.info(f"LR: {self.learning_rate}, Initial eta: {self.initial_eta}")
        logger.info(f"Gradient clipping: {self.max_grad_norm}")
        
        self.train()
        optimizer = self.configure_optimizers()
        
        # Setup learning rate scheduler
        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=self.warmup_steps)
        cosine_steps = max(1, num_steps - self.warmup_steps)
        print(cosine_steps)
        main_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_steps)
        scheduler = SequentialLR(optimizer, 
                               schedulers=[warmup_scheduler, main_scheduler],
                               milestones=[self.warmup_steps])

        step = 0
        while step < num_steps:
            optimizer.zero_grad()
            batch_loss = 0.0
            valid_accumulations = 0

            # Gradient accumulation
            for i in range(accum_steps):
                loss = self.training_step(step, num_steps)
                
                if torch.isfinite(loss) and loss.item() > 0:
                    (loss / accum_steps).backward()
                    batch_loss += loss.item()
                    valid_accumulations += 1

            if valid_accumulations == 0:
                logger.warning(f"No valid accumulations at step {step}")
                continue

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.max_grad_norm
            )
            self.grad_norms.append(grad_norm.item())

            # Optimizer step
            optimizer.step()
            scheduler.step()

            # Update metrics
            step += 1
            avg_loss = batch_loss / valid_accumulations
            
            # EMA smoothing
            if self.loss_ema is None:
                self.loss_ema = avg_loss
            else:
                self.loss_ema = self.ema_decay * self.loss_ema + (1 - self.ema_decay) * avg_loss
            
            self.losses.append(avg_loss)

            # Logging
            if step % 5 == 0 or step <= 20:
                current_lr = scheduler.get_last_lr()[0]
                logger.info(f"Step {step}/{num_steps} - Loss: {avg_loss:.6f} "
                          f"(EMA: {self.loss_ema:.6f}) - Eta: {self.eta:.3f} "
                          f"- LR: {current_lr:.2e} - Grad: {grad_norm:.4f}")

            # Validation/monitoring
            if step % self.validation_interval == 0:
                self.monitor_training(step)

        # Save model
        if save_path:
            self.save_model(save_path)

        # Plot results
        self.plot_training_curves()
        self.save_losses_json("training_losses.json")


    def monitor_training(self, step):
        """Monitor training stability"""
        if len(self.losses) < 10:
            return
            
        recent_losses = self.losses[-10:]
        loss_std = np.std(recent_losses)
        loss_mean = np.mean(recent_losses)
        
        if loss_std > loss_mean * 0.5:  # High variance
            logger.warning(f"High loss variance detected at step {step}: "
                         f"mean={loss_mean:.4f}, std={loss_std:.4f}")
        
        if len(self.grad_norms) > 5:
            recent_grads = self.grad_norms[-5:]
            if any(g > self.max_grad_norm * 0.8 for g in recent_grads):
                logger.warning(f"High gradient norms detected: {recent_grads}")

    def plot_training_curves(self):
        """Plot training curves"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Loss curve
        steps = range(1, len(self.losses) + 1)
        ax1.plot(steps, self.losses, alpha=0.7, label='Loss')
        if len(self.losses) > 10:
            # Moving average
            window = min(10, len(self.losses) // 5)
            ma_losses = np.convolve(self.losses, np.ones(window)/window, mode='valid')
            ma_steps = range(window, len(self.losses) + 1)
            ax1.plot(ma_steps, ma_losses, 'r-', label=f'MA({window})')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Gradient norms
        if self.grad_norms:
            ax2.plot(self.grad_norms, 'g-', alpha=0.7)
            ax2.axhline(y=self.max_grad_norm, color='r', linestyle='--', label='Clip threshold')
            ax2.set_xlabel('Step')
            ax2.set_ylabel('Gradient Norm')
            ax2.set_title('Gradient Norms')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Eta schedule
        if self.eta_values:
            ax3.plot(self.eta_values, 'purple', alpha=0.7)
            ax3.set_xlabel('Step')
            ax3.set_ylabel('Eta')
            ax3.set_title('Eta Schedule')
            ax3.grid(True, alpha=0.3)
        
        # Loss distribution
        ax4.hist(self.losses, bins=20, alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Loss Value')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Loss Distribution')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info("Training analysis plots saved to training_analysis.png")

    def save_model(self, save_path):
        """Save the fine-tuned model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'eta': self.eta,
            'losses': self.losses,
            'grad_norms': self.grad_norms,
            'eta_values': self.eta_values
        }, save_path)
        logger.info(f"Model saved to: {save_path}")


def main():
    """Example usage with stabilized training"""
    from genie.config import Config
    
    # Load configuration
    config = Config()
    
    # Define motif scaffolding files
    motif_dir = "coord_datasets/alpha_helix_dataset/alpha_helix_templates/"
    motif_scaffolding_files = [
        os.path.join(motif_dir, f)
        for f in os.listdir(motif_dir)
        if f.endswith(".pdb")
    ]
    
    logger.info(f"Found {len(motif_scaffolding_files)} motif files")
    
    # Initialize stabilized fine-tuner with more conservative settings
    fine_tuner = StabilizedGenieFineTuner(
        pretrained_model_path="results/base/checkpoints/epoch=40.ckpt",
        config=config,
        eta=0.3,
        learning_rate=2e-5,  # Even lower learning rate
        num_samples_per_step=2,  # Just 1 sample for maximum stability
        motif_scaffolding_files=motif_scaffolding_files,
        warmup_steps=50,  # Reduced warmup
        max_grad_norm=40,  # Lower gradient clipping
        eta_schedule="constant",
        validation_interval=5
    )
    
    # Run stabilized fine-tuning
    fine_tuner.fine_tune(
        num_steps=125,
        accum_steps=8,  # Much smaller accumulation
        save_path="stabilized_fine_tuned_model.pt"
    )





if __name__ == "__main__":
    main()