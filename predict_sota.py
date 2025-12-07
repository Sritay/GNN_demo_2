import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from typing import Optional

from data_collector import get_pbc_distances_sparse
from train_sota import SotaSchNet

# --- Configuration ---
TEST_FILE = 'data/test_data.xyz'
MODEL_PATH = 'best_sota_model.pth'
PLOT_OUTPUT = 'sota_parity.png'
CUTOFF = 5.0
Z_VALUE = 78

# Calibration Constant (Derived from training data)
# Represents the mean potential energy per atom of the reference system.
E_REF = -5.7335 

def parse_header(lines: list) -> Optional[torch.Tensor]:
    """
    Extracts simulation box dimensions from LAMMPS XYZ header.
    """
    try:
        xlo, xhi = map(float, lines[5].split())
        ylo, yhi = map(float, lines[6].split())
        zlo, zhi = map(float, lines[7].split())
        return torch.tensor([xhi-xlo, yhi-ylo, zhi-zlo], dtype=torch.float32)
    except (ValueError, IndexError):
        return None

def predict():
    """
    Runs inference on the test trajectory and generates a parity plot.
    """
    device = torch.device('cpu')
    
    # 1. Load Pre-trained Model
    print(f"Loading model architecture from {MODEL_PATH}...")
    model = SotaSchNet().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    # 2. Parse Test Data
    print(f"Reading test trajectory: {TEST_FILE}...")
    with open(TEST_FILE, 'r') as f:
        lines = f.readlines()

    n_atoms = int(lines[3])
    box_dims = parse_header(lines)
    
    if box_dims is None:
        raise ValueError("Failed to parse box dimensions from test file.")

    lines_per_frame = 9 + n_atoms
    total_frames = len(lines) // lines_per_frame
    print(f"Detected {total_frames} frames with {n_atoms} atoms each.")

    real_energies = []
    pred_energies = []

    # 3. Inference Loop
    for i in range(total_frames):
        start = i * lines_per_frame
        atom_lines = lines[start + 9 : start + lines_per_frame]
        
        # Parse frame
        data_arr = np.loadtxt(atom_lines)
        data_arr = data_arr[data_arr[:, 0].argsort()] # Sort by Atom ID
        
        pos = torch.tensor(data_arr[:, 2:5], dtype=torch.float32)
        target_e_total = float(np.sum(data_arr[:, 8]))
        
        # Construct Graph (Same pipeline as training)
        edge_index, edge_attr = get_pbc_distances_sparse(pos, box_dims, CUTOFF)
        z = torch.full((n_atoms,), Z_VALUE, dtype=torch.long)
        batch = torch.zeros(n_atoms, dtype=torch.long) # Single graph batch
        
        data = Data(
            z=z, 
            pos=pos, 
            edge_index=edge_index, 
            edge_attr=edge_attr, 
            batch=batch
        )
        
        # Predict
        with torch.no_grad():
            pred_delta = model(data)
            # Restore Total Energy: E_pred = Delta + (N * E_ref)
            pred_total = pred_delta.item() + (n_atoms * E_REF)
            
        real_energies.append(target_e_total)
        pred_energies.append(pred_total)

    # 4. Analysis & Visualization
    real_energies = np.array(real_energies)
    pred_energies = np.array(pred_energies)
    
    mae = np.mean(np.abs(real_energies - pred_energies))
    mae_per_atom = mae / n_atoms
    
    print("\n" + "="*40)
    print(f"SOTA MODEL EVALUATION (SchNet)")
    print("="*40)
    print(f"MAE Total:    {mae:.4f} eV")
    print(f"MAE Per Atom: {mae_per_atom:.4f} eV/atom")
    
    # Parity Plot
    plt.figure(figsize=(6, 6), dpi=100)
    plt.scatter(real_energies, pred_energies, alpha=0.6, color='crimson', label='SchNet Predictions')
    
    min_val = min(real_energies.min(), pred_energies.min())
    max_val = max(real_energies.max(), pred_energies.max())
    
    # Plot Identity Line
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Ideal Parity')
    
    plt.xlabel("Ground Truth Energy (eV)")
    plt.ylabel("Predicted Energy (eV)")
    plt.title(f"SOTA Validation (SchNet)\nMAE: {mae_per_atom:.4f} eV/atom")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(PLOT_OUTPUT)
    print(f"Validation plot saved to '{PLOT_OUTPUT}'")

if __name__ == "__main__":
    predict()
