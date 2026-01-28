import torch
import argparse
import numpy as np
from torch_geometric.utils import remove_self_loops, to_undirected
from torch_geometric.data import Data
import pickle

def reconstruct_and_average(
    data_saved: dict, 
    original_embeddings: np.ndarray
) -> dict:
   
    merge_tree = data_saved["merge_tree"]
    final_node_ids = data_saved["final_node_ids"]
    
    mapping = {}
    super_vectors = {}
    
    print(f"{len(final_node_ids)}supernodes...")
    
    for root_id in final_node_ids:
        stack = [root_id]
        leaves = []
        
        while stack:
            curr = stack.pop()
            if curr in merge_tree:
                left, right = merge_tree[curr]
                stack.append(right)
                stack.append(left)
            else:
                leaves.append(curr)
        
        mapping[root_id] = leaves
    

    for root_id, leaf_indices in mapping.items():

        mean_vector = np.mean(original_embeddings[leaf_indices], axis=0)
        
        super_vectors[root_id] = mean_vector.astype(np.float32)

    print("计算完成。")
    
    return {
        "mapping": mapping,
        "super_vectors": super_vectors
    }

def main():

    parser = argparse.ArgumentParser(description="")
    
    parser.add_argument("--graph_name", type=str, required=False, default='citeseer')
    parser.add_argument("--relax", type=str, required=False, default='1')
    parser.add_argument("--path_data", type=str, required=False, default='../dataset')
    parser.add_argument("--ratio", type=float, required=False, default=0.3)

    args = parser.parse_args()

    relax = args.relax
    graph_name = args.graph_name
    PATH_data = args.path_data
    ratio = args.ratio

    data = torch.load(f'{PATH_data}/{graph_name}.pt', weights_only=False)
    edge_index = data['edge_index']
    embeddings = data['feat'].numpy()


    if relax == '1':
        with open(f'../{graph_name}/new_info_{ratio}_rel.pkl', 'rb') as f:
            data_saved = pickle.load(f)
    else:
        with open(f'../{graph_name}/new_info_{ratio}.pkl', 'rb') as f:
            data_saved = pickle.load(f)
            
    info_all = reconstruct_and_average(data_saved, embeddings)
    old_sp2nodelist = info_all['mapping']
    super_vectors = info_all['super_vectors']

    x = []
    count = 0
    sp2nodelist = dict()
    for sp_id, node_list in old_sp2nodelist.items():
        sp2nodelist[count] = node_list
        x.append(torch.tensor(super_vectors[sp_id], dtype=torch.float32))
        count += 1

    
    node2sp = dict()
    for sp, node_list in sp2nodelist.items():
        for node in node_list:
            node2sp[node] = sp


    sp_edge_index_1 = []
    sp_edge_index_2 = []

    for node_i, node_j in zip(edge_index[0], edge_index[1]):
        sp_i = node2sp[node_i.item()]
        sp_j = node2sp[node_j.item()]
        if sp_i != sp_j:
            sp_edge_index_1.append(sp_i)
            sp_edge_index_2.append(sp_j)
    
    sp_edge_index = torch.stack([torch.tensor(sp_edge_index_1), torch.tensor(sp_edge_index_2)], dim=0).to(torch.int64)
    sp_edge_index = to_undirected(sp_edge_index)
    sp_edge_index, _ = remove_self_loops(sp_edge_index)
    
    x = torch.stack(x, dim=0).to(torch.float32)
    num_nodes = x.shape[0]

    sp_data = Data(x=x, edge_index=sp_edge_index, num_nodes=num_nodes)

    if relax == '1':
        torch.save(node2sp, f'../{graph_name}/node2sp_{ratio}_relax.pt')
        torch.save(sp_data, f'../{graph_name}/sp_data_{ratio}_relax.pt')
    else:
        torch.save(node2sp, f'../{graph_name}/node2sp_{ratio}.pt')
        torch.save(sp_data, f'../{graph_name}/sp_data_{ratio}.pt')

if __name__ == '__main__':
    main()
