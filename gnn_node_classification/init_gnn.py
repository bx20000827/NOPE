import torch
import argparse
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv
import torch_geometric.nn as nn
import torch.nn as tnn  
import os

from sklearn.metrics import (
    hamming_loss, f1_score, accuracy_score
)

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_classes, type: str, num_heads=4):
        super().__init__()
        gat_hidden_out = hidden_channels * num_heads 

        if type == 'GCN':
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, out_channels)
        elif type == 'GAT':
            self.conv1 = GATConv(in_channels, hidden_channels, heads=num_heads, dropout=0.5) 
            self.conv2 = GATConv(gat_hidden_out, out_channels, heads=num_heads, concat=False, dropout=0.5)
        elif type == 'SAGE':
            self.conv1 = SAGEConv(in_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, out_channels)
        elif type == 'GIN':
            
            self.conv1 = GINConv(
                tnn.Sequential(
                    tnn.Linear(in_channels, hidden_channels),
                    tnn.ReLU(),
                    tnn.Linear(hidden_channels, hidden_channels),
                    tnn.BatchNorm1d(hidden_channels), 
                ), train_eps=True 
            )
            
            self.conv2 = GINConv(
                tnn.Sequential(
                    tnn.Linear(hidden_channels, hidden_channels),
                    tnn.ReLU(),
                    tnn.Linear(hidden_channels, out_channels), 
                ), train_eps=True
            )
        else:
            raise ValueError(f"Unknown GNN type: {type}. Choose 'GCN', 'GAT', or 'SAGE'.")

        self.Classifier = tnn.Sequential(
            tnn.Linear(out_channels, num_classes)
        )

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.Classifier(x)
        return x

def train(model, optimizer, criterion, x, edge_index, train_nodes, train_y):
    model.train()
    optimizer.zero_grad()
    out = model(x, edge_index)
    out = out[train_nodes]
    loss = criterion(out, train_y)
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def evaluate(model, criterion, x, edge_index, nodes, y_true, graph_name):
    model.eval()
    out = model(x, edge_index)
    out = out[nodes]
    
    loss = criterion(out, y_true)
    
    out_np = out.cpu().numpy()
    y_true_np = y_true.cpu().numpy()
    
    metrics = {}
    
    if graph_name == 'book':
        probs = torch.sigmoid(out)
        preds = (probs > 0.5).float().cpu().numpy()
        metrics['score'] = f1_score(y_true_np, preds, average='weighted', zero_division=0) # F1
        metrics['hloss'] = hamming_loss(y_true_np, preds)
    else:
        preds = np.argmax(out_np, axis=1)
        metrics['score'] = accuracy_score(y_true_np, preds) # Acc
        metrics['f1'] = f1_score(y_true_np, preds, average='weighted', zero_division=0)
        
    return loss.item(), metrics

def main():
    graph_name = 'citeseer'
    PATH_data = '../dataset' 
    GNN_type = 'SAGE'
    set_seed(seed=42) # 42 114514 3407
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    patience = 50      
    patience_counter = 0 

    print(f"Loading data: {graph_name}...")
    data = torch.load(f'{PATH_data}/{graph_name}.pt', weights_only=False)
    
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
    edge_index = data['edge_index'].to(device)

    if graph_name == 'book':
        num_classes = y.shape[1]
        train_y = y[train_nodes].to(torch.float32).to(device)
        val_y = y[val_nodes].to(torch.float32).to(device)
        test_y = y[test_nodes].to(torch.float32).to(device)
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        num_classes = y.max().item() + 1
        train_y = y[train_nodes].to(torch.long).to(device)
        val_y = y[val_nodes].to(torch.long).to(device)
        test_y = y[test_nodes].to(torch.long).to(device)
        criterion = torch.nn.CrossEntropyLoss()

    in_dim = feat.shape[1]
    hidden_dim = 128
    out_dim = 64
    epochs = 4000

    model = GNN(in_dim, hidden_dim, out_dim, num_classes, GNN_type)
    model.to(device)
    x = feat.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=5e-2)
    
    save_dir = f'./best_model/{graph_name}/{GNN_type}'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'init_best_model.pth')
    
    best_val_loss = float('inf') 
    
    print(f"Start Training (Max Epochs: {epochs}, Patience: {patience})...")
    
    for epoch in range(epochs):
        train_loss = train(model, optimizer, criterion, x, edge_index, train_nodes, train_y)
        
        val_loss, val_metrics = evaluate(model, criterion, x, edge_index, val_nodes, val_y, graph_name)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0 
            torch.save(model.state_dict(), save_path)
        
            # print(f"Epoch {epoch}: Val Loss improved to {val_loss:.4f}. Model saved.")
        else:
            patience_counter += 1
            
        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Patience: {patience_counter}/{patience}')

        if patience_counter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch}. Validation loss hasn't improved for {patience} epochs.")
            break

    print("\nLoading best model for Final Testing...")
    model.load_state_dict(torch.load(save_path, weights_only=True, map_location=device))
    
    test_loss, test_metrics = evaluate(model, criterion, x, edge_index, test_nodes, test_y, graph_name)

    if graph_name == 'book':
        print(f"Final Test -> H-Loss: {test_metrics['hloss']:.4f}, F1-Score: {test_metrics['score']:.4f}")
    else:
        print(f"Final Test -> Acc: {test_metrics['score']:.4f}, F1-Score: {test_metrics['f1']:.4f}")

if __name__ == '__main__':
    main()