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
    create_np_features_from_motif_pdb,
    create_empty_np_features,
    convert_np_features_to_tensor,
    batchify_np_features
)


class GenieFineTuner(LightningModule):
    """
    Fine-tuning class for Genie 2 model that implements concept erasure/adjustment
    through double sampling and L2/MSE loss optimization.
    
    This follows the approach described in the concept erasure paper where:
    1. Sample the original model twice (conditioned and unconditioned)
    2. Compute target using: ε_θ*(x_t, t) - η[ε_θ*(x_t, c, t) - ε_θ*(x_t, t)]
    3. Optimize via L2 loss to make fine-tuned model match this target
    """

    def __init__(
        self,
        pretrained_model_path,
        config,
        eta=1.0,
        learning_rate=1e-5,
        max_length=256,
        num_samples_per_step=4,
        motif_pdb_paths=None
    ):
        """
        Args:
            pretrained_model_path: Path to the pretrained Genie 2 model
            config: Configuration object for the model
            eta: Power factor controlling the strength of concept adjustment
            learning_rate: Learning rate for fine-tuning
            max_length: Maximum sequence length for fine-tuning
            num_samples_per_step: Number of samples to generate per optimization step
            motif_pdb_paths: List of paths to motif PDB files for conditioning
        """
        super(GenieFineTuner, self).__init__()
        self.config = config
        self.eta = eta
        self.learning_rate = learning_rate
        self.max_length = max_length
        self.num_samples_per_step = num_samples_per_step
        self.motif_pdb_paths = motif_pdb_paths or []
        
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
        # This would need to be implemented based on your model loading logic
        # For now, creating a new model with the same config
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
        self.betas = get_betas(
            self.config.diffusion['n_timestep'],
            self.config.diffusion['schedule']
        ).to(self.device)

        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_cumprod_prev = torch.cat([
            torch.Tensor([1.]).to(self.device),
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

    def generate_conditioned_features(self, sequence_length):
        """
        Generate features for conditioned generation (with motif) using proper Genie 2 approach
        
        Args:
            sequence_length: Total sequence length
            
        Returns:
            features: Dictionary containing conditioned features
        """
        # Use real motif PDB if available, otherwise create synthetic motif
        if self.motif_pdb_paths:
            # Randomly select a motif PDB file
            motif_pdb_path = np.random.choice(self.motif_pdb_paths)
            try:
                # Create features from real motif PDB
                np_features = create_np_features_from_motif_pdb(motif_pdb_path)
                
                # If the motif is too long, truncate it
                if np_features['num_residues'] > sequence_length:
                    # Truncate to fit sequence length
                    np_features['aatype'] = np_features['aatype'][:sequence_length]
                    np_features['atom_positions'] = np_features['atom_positions'][:sequence_length]
                    np_features['residue_mask'] = np_features['residue_mask'][:sequence_length]
                    np_features['residue_index'] = np_features['residue_index'][:sequence_length]
                    np_features['chain_index'] = np_features['chain_index'][:sequence_length]
                    np_features['fixed_sequence_mask'] = np_features['fixed_sequence_mask'][:sequence_length]
                    np_features['fixed_structure_mask'] = np_features['fixed_structure_mask'][:sequence_length, :sequence_length]
                    np_features['fixed_group'] = np_features['fixed_group'][:sequence_length]
                    np_features['interface_mask'] = np_features['interface_mask'][:sequence_length]
                    np_features['num_residues'] = sequence_length
                    np_features['num_residues_per_chain'] = np.array([sequence_length])
                
                # Convert to tensor and batch
                features = convert_np_features_to_tensor(
                    batchify_np_features([np_features]), self.device
                )
                return features
                
            except Exception as e:
                print(f"Warning: Could not load motif PDB {motif_pdb_path}: {e}")
                print("Falling back to synthetic motif generation")
        
        # Fallback: Create synthetic motif features
        return self._create_synthetic_conditioned_features(sequence_length)
    
    def _create_synthetic_conditioned_features(self, sequence_length):
        """
        Create synthetic conditioned features as fallback
        """
        # Create empty features first
        np_features = create_empty_np_features([sequence_length])
        
        # Add synthetic motif (random positions)
        motif_length = min(np.random.randint(5, 20), sequence_length)
        motif_start = np.random.randint(0, sequence_length - motif_length + 1)
        motif_end = motif_start + motif_length
        
        # Set motif sequence and structure
        for i in range(motif_length):
            pos = motif_start + i
            aa_type = np.random.randint(0, 20)
            np_features['aatype'][pos, aa_type] = 1.0
            np_features['fixed_sequence_mask'][pos] = 1.0
            np_features['atom_positions'][pos] = np.random.randn(3) * 5.0  # Random coordinates
        
        # Set pairwise structure mask for motif
        motif_positions = np.arange(motif_start, motif_end)
        for i in motif_positions:
            for j in motif_positions:
                np_features['fixed_structure_mask'][i, j] = 1.0
        
        # Convert to tensor and batch
        features = convert_np_features_to_tensor(
            batchify_np_features([np_features]), self.device
        )
        return features

    def generate_unconditioned_features(self, sequence_length):
        """
        Generate features for unconditioned generation (no motif) using proper Genie 2 approach
        
        Args:
            sequence_length: Total sequence length
            
        Returns:
            features: Dictionary containing unconditioned features
        """
        # Create empty features using Genie 2 utilities
        np_features = create_empty_np_features([sequence_length])
        
        # Convert to tensor and batch
        features = convert_np_features_to_tensor(
            batchify_np_features([np_features]), self.device
        )
        return features

    def add_noise(self, features, timesteps):
        """
        Add noise to clean coordinates at given timesteps
        
        Args:
            features: Feature dictionary
            timesteps: [B] Timesteps for each sample
            
        Returns:
            noisy_features: Features with noisy coordinates
            noise: The actual noise that was added
        """
        coords = features['atom_positions']
        noise = torch.randn_like(coords) * features['residue_mask'].unsqueeze(-1)
        
        # Apply noise
        noisy_coords = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1) * coords + \
                      self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1) * noise
        
        # Create frames
        rots = compute_frenet_frames(
            noisy_coords,
            features['chain_index'],
            features['residue_mask']
        )
        ts = T(rots, noisy_coords)
        
        # Update features
        noisy_features = features.copy()
        noisy_features['atom_positions'] = noisy_coords
        
        return noisy_features, ts, noise

    def training_step(self, batch, batch_idx):
        """
        Single fine-tuning step following the concept erasure approach
        
        Args:
            batch: Batch of training data (motif information)
            batch_idx: Batch index
            
        Returns:
            loss: L2 loss between fine-tuned model and target
        """
        # Generate random sequence length
        sequence_length = torch.randint(50, self.max_length + 1, (1,)).item()
        
        # Sample timesteps
        timesteps = torch.randint(
            1, self.config.diffusion['n_timestep'] + 1,
            size=(self.num_samples_per_step,)
        ).to(self.device)
        
        total_loss = 0.0
        
        for i in range(self.num_samples_per_step):
            # Generate conditioned features (with motif) using proper Genie 2 approach
            conditioned_features = self.generate_conditioned_features(sequence_length)
            
            # Generate unconditioned features (no motif) using proper Genie 2 approach
            unconditioned_features = self.generate_unconditioned_features(sequence_length)
            
            # Add noise
            conditioned_features_noisy, ts_cond, noise_cond = self.add_noise(
                conditioned_features, timesteps[i:i+1]
            )
            unconditioned_features_noisy, ts_uncond, noise_uncond = self.add_noise(
                unconditioned_features, timesteps[i:i+1]
            )
            
            # Sample frozen model twice (conditioned and unconditioned)
            with torch.no_grad():
                # Conditioned prediction from frozen model
                frozen_cond_output = self.frozen_model(ts_cond, timesteps[i:i+1], conditioned_features_noisy)
                frozen_cond_noise = frozen_cond_output['z']
                
                # Unconditioned prediction from frozen model
                frozen_uncond_output = self.frozen_model(ts_uncond, timesteps[i:i+1], unconditioned_features_noisy)
                frozen_uncond_noise = frozen_uncond_output['z']
            
            # Compute target using the concept erasure formula
            # Target = ε_θ*(x_t, t) - η[ε_θ*(x_t, c, t) - ε_θ*(x_t, t)]
            guidance_vector = frozen_cond_noise - frozen_uncond_noise
            target_noise = frozen_uncond_noise - self.eta * guidance_vector
            
            # Sample fine-tuned model (conditioned)
            fine_tuned_output = self.model(ts_cond, timesteps[i:i+1], conditioned_features_noisy)
            fine_tuned_noise = fine_tuned_output['z']
            
            # Compute L2 loss
            loss = self.loss_fn(fine_tuned_noise, target_noise)
            total_loss += loss
        
        avg_loss = total_loss / self.num_samples_per_step
        
        # Logging
        self.log('fine_tune_loss', avg_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('eta', self.eta, on_step=True, on_epoch=True)
        
        return avg_loss

    def configure_optimizers(self):
        """Configure optimizer for fine-tuning"""
        return Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4
        )

    def fine_tune(self, num_epochs=100, save_path=None):
        """
        Run fine-tuning process
        
        Args:
            num_epochs: Number of epochs to fine-tune
            save_path: Path to save fine-tuned model
        """
        self.train()
        
        optimizer = self.configure_optimizers()
        
        # Create dummy dataset for training loop
        dummy_batch = [None] * 1000  # Dummy batches
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            
            for batch_idx, _ in enumerate(tqdm(dummy_batch, desc=f"Epoch {epoch+1}/{num_epochs}")):
                optimizer.zero_grad()
                
                loss = self.training_step(None, batch_idx)
                
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_epoch_loss = epoch_loss / len(dummy_batch)
            print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_epoch_loss:.6f}")
            
            # Save checkpoint
            if save_path and (epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join(save_path, f"fine_tuned_epoch_{epoch+1}.pt")
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch + 1,
                    'loss': avg_epoch_loss,
                    'eta': self.eta
                }, checkpoint_path)
                print(f"Saved checkpoint: {checkpoint_path}")

    def save_model(self, save_path):
        """Save the fine-tuned model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'eta': self.eta
        }, save_path)
        print(f"Fine-tuned model saved to: {save_path}")


def main():
    """Example usage of the fine-tuner"""
    from genie.config import Config
    
    # Load configuration
    config = Config()
    
    # Define motif PDB paths for conditioning (optional)
    motif_pdb_paths = [
        "data/design25/1prw.pdb",
        "data/design25/1qjg.pdb", 
        "data/design25/2kl8.pdb"
    ]
    
    # Initialize fine-tuner
    fine_tuner = GenieFineTuner(
        pretrained_model_path="path/to/pretrained/model.pt",
        config=config,
        eta=1.0,  # Adjust this to control concept adjustment strength
        learning_rate=1e-5,
        max_length=256,
        num_samples_per_step=4,
        motif_pdb_paths=motif_pdb_paths  # Use real motif PDBs for conditioning
    )
    
    # Run fine-tuning
    fine_tuner.fine_tune(
        num_epochs=100,
        save_path="fine_tuned_models"
    )
    
    # Save final model
    fine_tuner.save_model("fine_tuned_genie2.pt")


if __name__ == "__main__":
    main()
