import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# load base pretrained model
from genie.fine_tune import GenieFineTuner

# Path to the PDB file
pdb_path = "data/design25/1bdf.pdb"

# Initialize the fine-tuner (adjust model_path as needed)
pretrained_model_path = "../results/base/checkpoints/epoch=40.ckpt"
pretrained_model = GenieFineTuner(pretrained_model_path=pretrained_model_path)

# Generate conditioned features for the PDB file
features = pretrained_model.generate_conditioned_features(pdb_path)

# Print or process the features
print("Conditioned features for", pdb_path)
print(features)

# evaluate on the design25/1bdf.pdb