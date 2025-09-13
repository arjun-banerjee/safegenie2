import os
import json
import logging
from genie.config import Config
from fine_tune import StabilizedGenieFineTuner  # assuming your class code is in stabilized_fine_tuner.py

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_losses_json(fine_tuner, filename):
    """Save training metrics to JSON"""
    data = {
        "losses": fine_tuner.losses,
        "grad_norms": fine_tuner.grad_norms,
        "eta_values": fine_tuner.eta_values,
    }
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved training results to {filename}")

def main():
    # Load configuration
    config = Config()
    
    # Define motif scaffolding files
    motif_dir = "coord_datasets/prion_dataset/human_prion/"
    motif_scaffolding_files = [
        os.path.join(motif_dir, f)
        for f in os.listdir(motif_dir)
        if f.endswith(".pdb")
    ]
    
    logger.info(f"Found {len(motif_scaffolding_files)} motif files")
    
    # Sweep through eta values
    eta_values = [5, 10]
    for eta in eta_values:
        logger.info(f"=== Starting run with eta={eta} ===")
        
        fine_tuner = StabilizedGenieFineTuner(
            pretrained_model_path="results/base/checkpoints/epoch=40.ckpt",
            config=config,
            eta=eta,
            learning_rate=2e-5,
            num_samples_per_step=2,
            motif_scaffolding_files=motif_scaffolding_files,
            warmup_steps=50,
            max_grad_norm=40,
            eta_schedule="constant",
            validation_interval=5
        )
        
        fine_tuner.fine_tune(
            num_steps=250,
            accum_steps=8,
            save_path=f"prion_stabilized_fine_tuned_model_eta_{eta}.pt"
        )
        
        # Save training results to JSON
        save_losses_json(fine_tuner, f"prion_training_losses_eta_{eta}.json")

if __name__ == "__main__":
    main()
