import torch
import numpy as np
from torch_geometric.data import Data, InMemoryDataset
from typing import Tuple, Optional, List

# --- Constants ---
DATA_PATH = 'data/training_data.xyz'
PROCESSED_DIR = 'data_pyg'
BOX_SIZE = 11.76  # Simulation box length (Angstroms)
Z_VALUE = 78      # Atomic number for Platinum (Pt)
CUTOFF = 5.0      # Interaction cutoff radius (Angstroms)

def get_pbc_distances_sparse(
    pos: torch.Tensor, 
    box_dims: torch.Tensor, 
    cutoff: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes sparse graph connectivity and edge distances under Periodic Boundary Conditions (PBC).
    
    Implements the Minimum Image Convention (MIC) to ensure edges wrap correctly 
    across simulation box boundaries.

    Args:
        pos (torch.Tensor): Atomic positions of shape (N, 3).
        box_dims (torch.Tensor): Dimensions of the orthogonal simulation box (3,).
        cutoff (float): Radial cutoff distance for edge construction.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - edge_index (2, E): Sparse adjacency list.
            - edge_attr (E, 1): Radial distances for each edge.
    """
    # Broadcast positions to compute pairwise displacement (N, N, 3)
    # Note: For systems >10k atoms, use torch_cluster.radius_graph for efficiency.
    delta = pos.unsqueeze(0) - pos.unsqueeze(1)
    
    # Apply Minimum Image Convention
    box = box_dims.view(1, 1, 3)
    delta = delta - box * torch.round(delta / box)
    
    # Compute Euclidean distances
    dists = torch.sqrt((delta**2).sum(dim=2) + 1e-8)
    
    # Construct sparse graph based on cutoff
    # Mask out self-loops (dist ~ 0) and distant atoms
    mask = (dists < cutoff) & (dists > 0.01)
    
    row, col = torch.nonzero(mask, as_tuple=True)
    edge_index = torch.stack([row, col], dim=0)
    edge_attr = dists[row, col].unsqueeze(1)
    
    return edge_index, edge_attr

class PtDataset(InMemoryDataset):
    """
    PyTorch Geometric dataset for Platinum EAM potential training.
    
    Parses LAMMPS custom dump files and pre-computes graph topology 
    with PBC-corrected edges.
    """
    def __init__(self, root: str, xyz_file: str, transform: Optional[callable] = None):
        self.xyz_file = xyz_file
        super().__init__(root, transform, pre_transform=None)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self) -> List[str]:
        return [self.xyz_file]

    @property
    def processed_file_names(self) -> List[str]:
        return ['pt_graph_data.pt']

    def process(self):
        """
        Reads raw LAMMPS data, normalizes energy, constructs graphs, and saves to disk.
        """
        print(f"Processing raw data from {self.xyz_file}...")
        
        with open(self.xyz_file, 'r') as f:
            lines = f.readlines()

        # Parse header to determine system size
        try:
            n_atoms = int(lines[3])
        except (ValueError, IndexError):
            raise ValueError("Invalid LAMMPS dump format. Could not parse atom count.")
            
        lines_per_frame = 9 + n_atoms
        total_frames = len(lines) // lines_per_frame
        
        data_list = []
        all_energies = []
        box_dims = torch.tensor([BOX_SIZE, BOX_SIZE, BOX_SIZE], dtype=torch.float32)

        # Pass 1: Parse Frames and Collect Energies
        raw_frames = []
        for i in range(total_frames):
            start = i * lines_per_frame
            atom_lines = lines[start + 9 : start + lines_per_frame]
            
            # Load frame data
            data = np.loadtxt(atom_lines)
            data = data[data[:, 0].argsort()]  # Ensure consistent atom ordering by ID
            
            pos = torch.tensor(data[:, 2:5], dtype=torch.float32)
            forces = torch.tensor(data[:, 5:8], dtype=torch.float32)
            e_frame = np.sum(data[:, 8])
            
            raw_frames.append((pos, forces, e_frame))
            all_energies.append(e_frame)

        # Calculate Reference Energy (E_ref) for normalization
        mean_total_energy = np.mean(all_energies)
        e_ref = mean_total_energy / n_atoms
        print(f"Dataset Normalization: E_ref = {e_ref:.4f} eV/atom")

        # Pass 2: Construct Graph Objects
        print("Constructing sparse graphs with PBC...")
        for pos, forces, e_frame in raw_frames:
            # Normalize target energy (Delta E)
            e_delta = torch.tensor([e_frame - (n_atoms * e_ref)], dtype=torch.float32)
            
            # Compute topology
            edge_index, edge_dist = get_pbc_distances_sparse(pos, box_dims, CUTOFF)
            
            # Node features (Atomic Number Z)
            z = torch.full((n_atoms,), Z_VALUE, dtype=torch.long)
            
            data = Data(
                z=z, 
                pos=pos, 
                y=e_delta, 
                force=forces,
                edge_index=edge_index,
                edge_attr=edge_dist,
                n_atoms=n_atoms
            )
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print(f"Successfully saved {len(data_list)} graphs to {self.processed_paths[0]}")

if __name__ == "__main__":
    dataset = PtDataset(root=PROCESSED_DIR, xyz_file=DATA_PATH)
