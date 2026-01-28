import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
import os
import time
from sklearn.metrics import hamming_loss, f1_score, accuracy_score
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse import SparseTensor

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def precompute_sgc_features(x, edge_index, num_hops=2):

    print(f"[SGC] Starting pre-computation on CPU with {num_hops} hops...")
    start_t = time.time()
    
    num_nodes = x.size(0)
    
  
    print("[SGC] Normalizing adjacency matrix...")
    edge_index, edge_weight = gcn_norm(
        edge_index, num_nodes=num_nodes, add_self_loops=True, dtype=x.dtype
    )
    
    adj = SparseTensor(row=edge_index[0], col=edge_index[1], value=edge_weight,
                       sparse_sizes=(num_nodes, num_nodes))
    
    x_out = x
    for k in range(num_hops):
        x_out = adj.matmul(x_out)
        print(f"[SGC] Propagated hop {k+1}/{num_hops}")
        
    print(f"[SGC] Done in {time.time() - start_t:.2f}s. Shape: {x_out.shape}")
    return x_out

class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_classes, dropout=0.5):
        super().__init__()
        self.dropout_p = dropout
        
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        
        self.classifier = nn.Linear(out_channels, num_classes)

    def forward(self, x):
        # Layer 1
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        
        # Layer 2
        x = self.lin2(x)

        # Classifier
        x = self.classifier(x)
        return x

def train(model, optimizer, criterion, x_sgc, train_nodes, train_y):
    model.train()
    optimizer.zero_grad()
    
  
    out = model(x_sgc[train_nodes])
    loss = criterion(out, train_y)
    
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def evaluate(model, criterion, x_sgc, nodes, y_true, graph_name):
    model.eval()
    out = model(x_sgc[nodes])
    
    loss = criterion(out, y_true)
    
    out_np = out.cpu().numpy()
    y_true_np = y_true.cpu().numpy()
    
    metrics = {}
    
    if graph_name == 'book': # Multi-label
        probs = torch.sigmoid(out)
        preds = (probs > 0.5).float().cpu().numpy()
        metrics['score'] = f1_score(y_true_np, preds, average='weighted', zero_division=0) # F1
        metrics['hloss'] = hamming_loss(y_true_np, preds)
    else: # Single-label
        preds = np.argmax(out_np, axis=1)
        metrics['score'] = accuracy_score(y_true_np, preds) # Acc
        metrics['f1'] = f1_score(y_true_np, preds, average='weighted', zero_division=0)
        
    return loss.item(), metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='products_all')
    parser.add_argument('--path_data', type=str, default='')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--hops', type=int, default=2, help='K hops for SGC')
    args = parser.parse_args()

    graph_name = args.dataset
    set_seed(42)
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}, Dataset: {graph_name}, SGC Hops: {args.hops}")

    patience = 50       
    patience_counter = 0 

    print(f"Loading data: {graph_name}...")
    data = torch.load(f'{args.path_data}/dataset/{graph_name}.pt', weights_only=False)
    
    train_nodes = data['train_mask'].nonzero().squeeze().tolist()
    test_nodes = data['test_mask'].nonzero().squeeze().tolist()
    
    if 'val_mask' in data:
        val_nodes = data['val_mask'].nonzero().squeeze().tolist()
        print(f"Validation set found: {len(val_nodes)} nodes.")
    else:
        print("Warning: No 'val_mask' found. Using 'test_mask' for validation.")
        val_nodes = test_nodes

    print(f"Train node: {len(data['train_mask'].nonzero().squeeze())}")
    print(f"Validation node: {len(data['val_mask'].nonzero().squeeze())}")
    print(f"Test node: {len(data['test_mask'].nonzero().squeeze())}")

    feat = data['feat'].to(torch.float32)
    y = data['y']
    edge_index = data['edge_index'] 


    print("Pre-computing SGC features on CPU...")
    x_sgc = precompute_sgc_features(feat, edge_index, num_hops=args.hops)
    
    print("Moving SGC features to GPU...")
    x_sgc = x_sgc.to(device)

    if graph_name == 'book':
        num_classes = y.shape[1]
        train_y = y[train_nodes].to(torch.float32).to(device)
        val_y = y[val_nodes].to(torch.float32).to(device)
        test_y = y[test_nodes].to(torch.float32).to(device)
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        if y.dim() > 1 and y.shape[1] == 1: y = y.squeeze()
        num_classes = int(y.max().item()) + 1
        train_y = y[train_nodes].to(torch.long).to(device)
        val_y = y[val_nodes].to(torch.long).to(device)
        test_y = y[test_nodes].to(torch.long).to(device)
        criterion = torch.nn.CrossEntropyLoss()

    in_dim = feat.shape[1]
    hidden_dim = 128
    out_dim = 64
    epochs = 4000

    model = MLP(in_dim, hidden_dim, out_dim, num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=5e-2)
    
    save_dir = f'./best_model/{graph_name}/SGC_MLP'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'init_best_model.pth')
    
    best_val_loss = float('inf') 
    
    print(f"Start Training MLP (Max Epochs: {epochs}, Patience: {patience})...")
    
    for epoch in range(epochs):
        train_loss = train(model, optimizer, criterion, x_sgc, train_nodes, train_y)
        
        val_loss, val_metrics = evaluate(model, criterion, x_sgc, val_nodes, val_y, graph_name)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0 
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            
        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Patience: {patience_counter}/{patience}')

        if patience_counter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch}. Validation loss hasn't improved for {patience} epochs.")
            break

    print("\nLoading best model for Final Testing...")
    model.load_state_dict(torch.load(save_path, weights_only=True, map_location=device))
    
    test_loss, test_metrics = evaluate(model, criterion, x_sgc, test_nodes, test_y, graph_name)

    if graph_name == 'book':
        print(f"Final Test -> H-Loss: {test_metrics['hloss']:.4f}, F1-Score: {test_metrics['score']:.4f}")
    else:
        print(f"Final Test -> Acc: {test_metrics['score']:.4f}, F1-Score: {test_metrics['f1']:.4f}")

if __name__ == '__main__':
    main()