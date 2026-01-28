import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, add_self_loops
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse import SparseTensor
import numpy as np
import os
import argparse
import time
from sklearn.metrics import hamming_loss, f1_score, accuracy_score
from torch.utils.data import DataLoader, TensorDataset

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def precompute_sgc_features(x, edge_index, num_hops=2):

    print(f"   [SGC] Starting pre-computation on CPU with {num_hops} hops...")
    start_t = time.time()
    
    num_nodes = x.size(0)
    

    print("   [SGC] Normalizing adjacency matrix...")
    edge_index, edge_weight = gcn_norm(
        edge_index, num_nodes=num_nodes, add_self_loops=True, dtype=x.dtype
    )
    
    adj = SparseTensor(row=edge_index[0], col=edge_index[1], value=edge_weight,
                       sparse_sizes=(num_nodes, num_nodes))
    
    x_out = x
    for k in range(num_hops):
        x_out = adj.matmul(x_out)
        print(f"   [SGC] Propagated hop {k+1}/{num_hops}")
        
    print(f"   [SGC] Done in {time.time() - start_t:.2f}s. Shape: {x_out.shape}")
    return x_out

class SGC_MLP_Model(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_classes, dropout=0.5):
        super().__init__()
        
        self.mlp_local = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )
        
  
        self.lin_global = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.classifier = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, num_classes)
        )
        
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, batch_x, batch_sgc_feat):
  
        local_emb = self.mlp_local(batch_x)
        global_emb = self.lin_global(batch_sgc_feat)
        
        combined = torch.cat([local_emb, global_emb], dim=1)
        out = self.classifier(combined)
        return out

def train_epoch(model, loader, feat, sgc_feat, node2sp, labels, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for batch_idx in loader:
        batch_x = feat[batch_idx].to(device)
        batch_y = labels[batch_idx].to(device)

        batch_sp_ids = node2sp[batch_idx]
        batch_sgc = sgc_feat[batch_sp_ids].to(device)
        
        optimizer.zero_grad()
        out = model(batch_x, batch_sgc)
        loss = criterion(out, batch_y)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch_idx.size(0)
        
    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, feat, sgc_feat, node2sp, labels, criterion, device, is_multilabel):
    model.eval()
    preds = []
    targets = []
    total_loss = 0
    
    for batch_idx in loader:
        batch_x = feat[batch_idx].to(device)
        batch_sp_ids = node2sp[batch_idx]
        batch_sgc = sgc_feat[batch_sp_ids].to(device)
        batch_y = labels[batch_idx] # Keep on CPU for metrics
        
        out = model(batch_x, batch_sgc)
        
        # Loss Calculation
        if is_multilabel:
            loss = criterion(out, batch_y.to(device).float())
        else:
            loss = criterion(out, batch_y.to(device).long())
        total_loss += loss.item() * batch_idx.size(0)
        
        preds.append(out.cpu())
        targets.append(batch_y)
        
    avg_loss = total_loss / len(loader.dataset)
    
    out_all = torch.cat(preds, dim=0)
    y_true = torch.cat(targets, dim=0).numpy()
    
    if is_multilabel:
        out_sig = torch.sigmoid(out_all).numpy()
        y_pred = (out_sig > 0.5).astype(float)
        m1 = hamming_loss(y_true, y_pred)
        m2 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    else:
        y_pred = torch.argmax(out_all, dim=1).numpy()
        m1 = accuracy_score(y_true, y_pred)
        m2 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
    return avg_loss, m1, m2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='products_all')
    parser.add_argument('--ratio', type=float, default=0.3)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--path_data', type=str, default='../dataset')
    parser.add_argument('--relax', action='store_true', default=True)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--hops', type=int, default=2, help='K hops for SGC')
    args = parser.parse_args()

    set_seed(42) # 114514 3407
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Config: {args.dataset} | Device: {device} | SGC Hops: {args.hops}")

    data_path = f'{args.path_data}/dataset/{args.dataset}.pt'
    print(f"Loading raw data from {data_path}...")
    data = torch.load(data_path, weights_only=False)
    
    feat = data['feat'].float()
    y = data['y']
    
    train_idx = data['train_mask'].nonzero().squeeze()
    test_idx = data['test_mask'].nonzero().squeeze()
    val_idx = data['val_mask'].nonzero().squeeze() if 'val_mask' in data else test_idx
    
    is_multilabel = (args.dataset == 'book')
    if is_multilabel:
        num_classes = y.shape[1]
        labels = y.float()
        criterion = nn.BCEWithLogitsLoss()
    else:
        if y.dim() > 1 and y.shape[1] == 1: y = y.squeeze()
        num_classes = int(y.max().item() + 1)
        labels = y.long()
        criterion = nn.CrossEntropyLoss()
        
    base_aux_path = f'./{args.dataset}'
    suffix = '_relax' if args.relax else ''
    print(f"Loading aux data{suffix}...")
    
    print(f"Loading aux data with suffix: '{suffix}'")
    # init_node2sp = torch.load(f'{base_aux_path}/node2sp{suffix}_{args.ratio}.pt', weights_only=False)
    # sp_data = torch.load(f'{base_aux_path}/sp_data{suffix}_{args.ratio}.pt', weights_only=False)
    init_node2sp = torch.load(f'{base_aux_path}/node2sp_{args.ratio}{suffix}.pt', weights_only=False)
    sp_data = torch.load(f'{base_aux_path}/sp_data_{args.ratio}{suffix}.pt', weights_only=False)


    if isinstance(init_node2sp, dict):
        node2sp = list(init_node2sp.values())
        node2sp = torch.tensor(node2sp, dtype=torch.long)
    else:
        node2sp = init_node2sp.clone().detach().to(torch.long)


    print("Pre-computing SGC features...")
    sp_sgc_features = precompute_sgc_features(sp_data.x, sp_data.edge_index, num_hops=args.hops)
    
   
    print("Moving SGC features to GPU for fast training...")
    sp_sgc_features = sp_sgc_features.to(device)


    train_loader = DataLoader(train_idx, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_idx, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_idx, batch_size=args.batch_size, shuffle=False)

    in_dim = feat.shape[1]
    model = SGC_MLP_Model(in_dim=in_dim, hidden_dim=128, out_dim=64, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    save_dir = f'{base_aux_path}/SGC_MLP'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'best_model_{args.ratio}.pth')
    
    best_val_loss = float('inf')
    patience_cnt = 0
    
    print("\nStart Training (Pure MLP Speed)...")
    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, feat, sp_sgc_features, node2sp, labels, optimizer, criterion, device)
        val_loss, val_m1, val_m2 = evaluate(model, val_loader, feat, sp_sgc_features, node2sp, labels, criterion, device, is_multilabel)
        
        epoch_time = time.time() - t0
        
        if epoch % 1 == 0:
            metric_name = "H-Loss" if is_multilabel else "Acc"
            print(f'Ep {epoch:03d} | Time: {epoch_time:.2f}s | Train Loss: {train_loss:.4f} | '
                  f'Val Loss: {val_loss:.4f} | Val {metric_name}: {val_m1:.4f} | F1: {val_m2:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_cnt = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print("Loading best model for testing...")
    model.load_state_dict(torch.load(save_path, weights_only=True, map_location=device))
    _, test_m1, test_m2 = evaluate(model, test_loader, feat, sp_sgc_features, node2sp, labels, criterion, device, is_multilabel)
    
    metric_name = "H-Loss" if is_multilabel else "Acc"
    print(f"Final Test -> {metric_name}: {test_m1:.4f}, F1-Score: {test_m2:.4f}")

if __name__ == '__main__':
    main()