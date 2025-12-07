import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models.schnet import GaussianSmearing, InteractionBlock, ShiftedSoftplus
from torch_scatter import scatter_add

from data_collector import PtDataset

# --- Hyperparameters ---
DATA_ROOT = 'data_pyg'
MODEL_SAVE_PATH = 'best_sota_model.pth'
BATCH_SIZE = 4
LEARNING_RATE = 5e-4
EPOCHS = 100
HIDDEN_CHANNELS = 64
NUM_FILTERS = 64
NUM_INTERACTIONS = 3
CUTOFF = 5.0

class SotaSchNet(nn.Module):
    """
    SchNet implementation adapted for explicit Periodic Boundary Conditions (PBC).
    
    Unlike standard implementations that recompute edges dynamically, this model 
    accepts pre-computed edge attributes to strictly enforce the Minimum Image Convention.
    """
    def __init__(self):
        super(SotaSchNet, self).__init__()
        
        # 1. Atomic Embedding (Z -> Hidden Vector)
        self.embedding = nn.Embedding(100, HIDDEN_CHANNELS)
        
        # 2. RBF Expansion
        self.distance_expansion = GaussianSmearing(0.0, CUTOFF, NUM_FILTERS)
        
        # 3. Interaction Blocks (Continuous Convolutions)
        self.interactions = nn.ModuleList()
        for _ in range(NUM_INTERACTIONS):
            block = InteractionBlock(HIDDEN_CHANNELS, NUM_FILTERS, NUM_FILTERS, CUTOFF)
            self.interactions.append(block)
            
        # 4. Output Heads
        self.lin1 = nn.Linear(HIDDEN_CHANNELS, HIDDEN_CHANNELS // 2)
        self.act = ShiftedSoftplus()
        self.lin2 = nn.Linear(HIDDEN_CHANNELS // 2, 1)

    def forward(self, data):
        """
        Forward pass with explicit edge handling.
        """
        # Node embeddings
        h = self.embedding(data.z)
        
        # Expand pre-computed PBC distances using Gaussian Basis
        edge_weight = data.edge_attr.view(-1)
        edge_attr = self.distance_expansion(edge_weight)
        
        # Message Passing
        for interaction in self.interactions:
            h = interaction(h, data.edge_index, edge_weight, edge_attr)
            
        # Readout
        h = self.lin1(h)
        h = self.act(h)
        h = self.lin2(h)
        
        # Global Pooling (Sum over atoms per graph)
        out = scatter_add(h, data.batch, dim=0)
        
        return out.view(-1)

def train():
    """
    Main training loop.
    """
    # Force CPU for small-scale serial job (optimal for <1000 atoms)
    device = torch.device('cpu')
    print(f"Initializing training on {device}...")

    # Load Dataset
    dataset = PtDataset(root=DATA_ROOT, xyz_file='data/training_data.xyz')
    
    # 80/20 Train-Validation Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Dataset Split: {len(train_dataset)} Training | {len(val_dataset)} Validation")
    
    # Initialize Model & Optimizer
    model = SotaSchNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
    print("\n--- Starting Training Loop ---")
    min_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        # Training Step
        model.train()
        train_loss = 0.0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            pred_e = model(batch)
            loss = criterion(pred_e, batch.y)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation Step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred_e = model(batch)
                loss = criterion(pred_e, batch.y)
                val_loss += loss.item()

        avg_train = train_loss / len(loader)
        avg_val = val_loss / len(val_loader)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d}: Train MSE = {avg_train:.4f} | Val MSE = {avg_val:.4f}")
            
        # Save Best Model
        if avg_val < min_val_loss:
            min_val_loss = avg_val
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print(f"Training complete. Best model weights saved to '{MODEL_SAVE_PATH}'.")

if __name__ == "__main__":
    train()
