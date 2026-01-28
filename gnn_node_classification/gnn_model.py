import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv
import numpy as np
import os
import argparse
from sklearn.metrics import hamming_loss, f1_score, accuracy_score

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class GNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, gnn_type: str, num_heads=4, dropout=0.5):
        super().__init__()
        self.dropout_ratio = dropout
        
        gat_hidden_out = hidden_channels * num_heads 

        if gnn_type == 'GCN':
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, out_channels)
        elif gnn_type == 'GAT':
            self.conv1 = GATConv(in_channels, hidden_channels, heads=num_heads, dropout=dropout)
            self.conv2 = GATConv(gat_hidden_out, out_channels, heads=num_heads, concat=False, dropout=dropout)
        elif gnn_type == 'SAGE':
            self.conv1 = SAGEConv(in_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, out_channels)

        elif gnn_type == 'GIN':
  
            self.conv1 = GINConv(
                nn.Sequential(
                    nn.Linear(in_channels, hidden_channels),
                    nn.ReLU(),
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.BatchNorm1d(hidden_channels), 
                ), train_eps=True 
            )
            
            self.conv2 = GINConv(
                nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.ReLU(),
                    nn.Linear(hidden_channels, out_channels), 
                ), train_eps=True
            )

        else:
            raise ValueError(f"Unknown GNN type: {gnn_type}")

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class GNNClassifier(nn.Module):
    def __init__(self, feature_dim, in_channels, hidden_channels, out_channels, gnn_type, num_classes, dropout=0.5):
        super().__init__()
        
        self.gnn = GNN(in_channels, hidden_channels, out_channels, gnn_type, dropout=dropout)
        
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_channels, out_channels)
        )

        self.classifier = nn.Sequential(
            nn.Linear(2 * out_channels, out_channels),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(out_channels, num_classes)
        )
        
        self.init_parameters()

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, edge_index, batch_sp_idx, batch_extra_feat):
 
        global_repr = self.gnn(x, edge_index)
        selected_repr = global_repr[batch_sp_idx]
        local_repr = self.mlp(batch_extra_feat)
        combined = torch.cat([selected_repr, local_repr], dim=1)
        out = self.classifier(combined)
        return out

def train(model, optimizer, criterion, x, edge_index, train_sp, train_x, train_y):
    model.train()
    optimizer.zero_grad()
    
    out = model(x, edge_index, train_sp, train_x)
    loss = criterion(out, train_y)
    
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def evaluate(model, criterion, x, edge_index, eval_sp, eval_x, eval_y_cpu, device, is_multilabel):
    model.eval()
    
    out = model(x, edge_index, eval_sp, eval_x)
    
    if is_multilabel:
        target_for_loss = eval_y_cpu.to(device) # float
    else:
        target_for_loss = eval_y_cpu.to(device).long() # long
        
    loss = criterion(out, target_for_loss).item()
    
    out = out.cpu()
    
    if is_multilabel:
        out = torch.sigmoid(out)
        preds = (out.numpy() > 0.5).astype(float)
        y_true = eval_y_cpu.numpy()
        
        hloss = hamming_loss(y_true, preds)
        f1 = f1_score(y_true, preds, average='weighted', zero_division=0)
        return loss, hloss, f1
    else:
        preds = torch.argmax(out, dim=1).numpy()
        y_true = eval_y_cpu.numpy()
        
        acc = accuracy_score(y_true, preds)
        f1 = f1_score(y_true, preds, average='weighted', zero_division=0)
        return loss, acc, f1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='book', help='Dataset name')
    parser.add_argument('--gnn_type', type=str, default='SAGE', choices=['GCN', 'GAT', 'SAGE', 'GIN'])
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument("--path_data", type=str, required=False, default='../dataset')
    parser.add_argument('--ratio', type=float, default=0.3, help='Coarsening ratio')
    parser.add_argument('--relax', action='store_true', default=True)
    parser.add_argument('--patience', type=int, default=50, help='Early stopping patience')
    args = parser.parse_args()

    set_seed(42) # 114514 3407
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    device = 'cuda:7' 
    print(f"Using device: {device}, Dataset: {args.dataset}, GNN: {args.gnn_type}, Relax: {args.relax}")

    data_path = f'{args.path_data}/{args.dataset}.pt'
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        return

    data = torch.load(data_path, weights_only=False)
    
    train_nodes = data['train_mask'].nonzero().squeeze()
    test_nodes = data['test_mask'].nonzero().squeeze()
    

    if 'val_mask' in data:
        val_nodes = data['val_mask'].nonzero().squeeze()
        print(f"Validation set found: {val_nodes.shape[0]} nodes.")
    else:
        print("Warning: 'val_mask' not found. Using 'test_mask' as validation.")
        val_nodes = test_nodes

    print(f"Train node: {len(data['train_mask'].nonzero().squeeze())}")
    print(f"Validation node: {len(data['val_mask'].nonzero().squeeze())}")
    print(f"Test node: {len(data['test_mask'].nonzero().squeeze())}")

    feat = data['feat'].to(torch.float32)
    y = data['y']

    is_multilabel = (args.dataset == 'book')
    
    if is_multilabel:
        num_classes = y.shape[1]
        labels = y.to(torch.float32)
        criterion = nn.BCEWithLogitsLoss()
    else:
        if y.dim() > 1 and y.shape[1] == 1:
            y = y.squeeze()
        num_classes = int(y.max().item() + 1)
        labels = y.to(torch.long) 
        criterion = nn.CrossEntropyLoss()

    base_aux_path = f'./{args.dataset}'
    suffix = '_relax' if args.relax else ''
    
    print(f"Loading aux data with suffix: '{suffix}'")
    init_node2sp = torch.load(f'{base_aux_path}/node2sp_{args.ratio}{suffix}.pt', weights_only=False)
    sp_data = torch.load(f'{base_aux_path}/sp_data_{args.ratio}{suffix}.pt', weights_only=False)

    if isinstance(init_node2sp, dict):
        node2sp = list(init_node2sp.values())
        node2sp = torch.tensor(node2sp, dtype=torch.long)
    else:
        node2sp = init_node2sp.clone().detach().to(torch.long)

    train_x = feat[train_nodes].to(device)
    val_x = feat[val_nodes].to(device)
    test_x = feat[test_nodes].to(device)
    
    train_y = labels[train_nodes].to(device)
    val_y_cpu = labels[val_nodes]
    test_y_cpu = labels[test_nodes] 

    train_sp = node2sp[train_nodes].to(device)
    val_sp = node2sp[val_nodes].to(device)
    test_sp = node2sp[test_nodes].to(device)

    x_global = sp_data.x.to(device)
    edge_index_global = sp_data.edge_index.to(device)

    in_dim = feat.shape[1]
    hidden_dim = 128
    out_dim = 64
    epochs = 2000

    model = GNNClassifier(
        feature_dim=in_dim,
        in_channels=in_dim,
        hidden_channels=hidden_dim,
        out_channels=out_dim,
        gnn_type=args.gnn_type,
        num_classes=num_classes
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    save_dir = f'{base_aux_path}/{args.gnn_type}'
    os.makedirs(save_dir, exist_ok=True)
    
    if args.relax:
        save_path = os.path.join(save_dir, f'best_model_{args.ratio}_rel.pth')
    else:
        save_path = os.path.join(save_dir, f'best_model_{args.ratio}.pth')
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"Start training with patience={args.patience}...")
    
    for epoch in range(epochs):
        train_loss = train(model, optimizer, criterion, x_global, edge_index_global, train_sp, train_x, train_y)
        
        val_loss, val_m1, val_m2 = evaluate(
            model, criterion, x_global, edge_index_global, val_sp, val_x, val_y_cpu, device, is_multilabel
        )
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0 
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1

        if epoch % 10 == 0:
            if is_multilabel:
                print(f'Epoch: {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val H-Loss: {val_m1:.4f}')
            else:
                print(f'Epoch: {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_m1:.4f}')

        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered at epoch {epoch}. (No improvement for {args.patience} epochs)")
            break

    print(f"\nLoading best model from {save_path} for Final Testing...")
    model.load_state_dict(torch.load(save_path, weights_only=True, map_location=device))
    
    test_loss, test_m1, test_m2 = evaluate(
        model, criterion, x_global, edge_index_global, test_sp, test_x, test_y_cpu, device, is_multilabel
    )

    if is_multilabel:
        print(f'Final Test -> Hamming Loss: {test_m1:.4f}, F1-Score: {test_m2:.4f}')
    else:
        print(f'Final Test -> Acc: {test_m1:.4f}, F1-Score: {test_m2:.4f}')

if __name__ == '__main__':
    main()