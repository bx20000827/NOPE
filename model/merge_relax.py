import heapq
import numpy as np
import time
from typing import Dict, List, Tuple, Set, Optional
import os
import torch


class GraphDataManager:

    def __init__(self, initial_embeddings: np.ndarray, initial_sizes: np.ndarray, max_nodes: int):
        self.dim = initial_embeddings.shape[1]
        self.max_nodes = max_nodes
        self.current_max_id = initial_embeddings.shape[0] - 1
        

        self.vectors = np.zeros((max_nodes, self.dim), dtype=np.float32)
        self.sizes = np.zeros(max_nodes, dtype=np.float32)
        self.degrees = np.zeros(max_nodes, dtype=np.float32)
        
     
        n = initial_embeddings.shape[0]
        self.vectors[:n] = initial_embeddings
        self.sizes[:n] = initial_sizes
        
        self.adj: Dict[int, Set[int]] = {}
        self.active_nodes = set(range(n))

    def add_node(self, new_id: int, vector: np.ndarray, size: float, neighbors: Set[int]):
        self.vectors[new_id] = vector
        self.sizes[new_id] = size
        self.adj[new_id] = neighbors
        self.degrees[new_id] = len(neighbors)
        self.active_nodes.add(new_id)

    def remove_node(self, node_id: int):
        if node_id in self.adj:
            del self.adj[node_id]
        self.active_nodes.discard(node_id)

    def update_degree(self, node_id: int, new_degree: int):
        self.degrees[node_id] = new_degree


def calculate_lm2_loss_fast(
    p_ids: List[int],
    q_ids: List[int],
    graph: GraphDataManager
) -> List[float]:

    if not p_ids:
        return []
    
    p_idx = np.array(p_ids, dtype=np.int32)
    q_idx = np.array(q_ids, dtype=np.int32)
    
    p_sizes = graph.sizes[p_idx]
    q_sizes = graph.sizes[q_idx]
    
    p_vecs = graph.vectors[p_idx]
    q_vecs = graph.vectors[q_idx]
    

    weights = (p_sizes * q_sizes) / (p_sizes + q_sizes + 1e-8)
    
    diffs = p_vecs - q_vecs
    dists_sq = np.sum(diffs**2, axis=1)

    
    deg_p = graph.degrees[p_idx]
    deg_q = graph.degrees[q_idx]
    
    batch_size = len(p_ids)
    intersection_lens = np.zeros(batch_size, dtype=np.float32)
    
    adj = graph.adj
    for i in range(batch_size):
        pid, qid = p_ids[i], q_ids[i]
        
        set_p = adj.get(pid, set())
        set_q = adj.get(qid, set())
        
        if len(set_p) < len(set_q):
            intersection_lens[i] = len(set_p.intersection(set_q))
        else:
            intersection_lens[i] = len(set_q.intersection(set_p))

    union_sizes = deg_p + deg_q - intersection_lens
    

    nr_sizes = union_sizes - 2.0
    nr_sizes = np.maximum(nr_sizes, 0)
    
    losses = weights * nr_sizes * dists_sq
    return losses.tolist()


def initialize_graph_edges(
    edge_index: np.ndarray, 
    num_nodes: int,
    graph: GraphDataManager
) -> Set[Tuple[int, int]]:
    
    rows, cols = edge_index
    all_edges = set()

    for u, v in zip(rows, cols):
        u, v = int(u), int(v)
        if u == v: continue
        
        if u not in graph.adj: graph.adj[u] = set()
        if v not in graph.adj: graph.adj[v] = set()
        
        graph.adj[u].add(v)
        graph.adj[v].add(u)
        
        all_edges.add(tuple(sorted((u, v))))

    for i in range(num_nodes):
        if i in graph.adj:
            graph.update_degree(i, len(graph.adj[i]))
            
    return all_edges


def hierarchical_merge_rel(
    embeddings, 
    edge_index, 
    n_nodes: int, 
    merge_ratio: float = 0.5, 
    verbose: bool = True,
    logger = None
):
    if verbose:
        try:
            import mkl
            if logger: logger.info(f"MKL Max Threads: {mkl.get_max_threads()}")
        except: pass
    
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    else:
        embeddings = np.array(embeddings, dtype=np.float32)
    
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.detach().cpu().numpy()
    
    target = max(1, int(round(n_nodes * merge_ratio)))
    
    start_time = time.time()


    max_possible_nodes = n_nodes * 2 
    initial_sizes = np.ones(n_nodes, dtype=np.float32)
    
    graph = GraphDataManager(embeddings, initial_sizes, max_possible_nodes)
    
    if verbose and logger:
        logger.info(f"Graph Manager initialization complete. Pre-allocated matrix size: {graph.vectors.shape}")

    all_edges = initialize_graph_edges(edge_index, n_nodes, graph)
    
    heap = []
    if all_edges:
        p_ids, q_ids = zip(*all_edges)
        initial_losses = calculate_lm2_loss_fast(list(p_ids), list(q_ids), graph)
        
        for loss, p_id, q_id in zip(initial_losses, p_ids, q_ids):
            heapq.heappush(heap, (loss, p_id, q_id))
            
    merge_tree: Dict[int, Tuple[int, int]] = {}
    merge_round = 0
    next_node_id = n_nodes
    
    def log_status():
        if logger:
            logger.info(f"Round {merge_round}: ActiveNodes={len(graph.active_nodes)}, "
                        f"HeapSize={len(heap)}")

    if verbose and logger:
        logger.info(f"Start compressing: {n_nodes} -> {target}")

    while len(graph.active_nodes) > target:
        
        valid_pair = False
        loss_val = float('inf')
        u_id, v_id = -1, -1
        
        while heap:
            loss_val, u, v = heapq.heappop(heap)
            if (u in graph.active_nodes and v in graph.active_nodes and 
                v in graph.adj.get(u, set())):
                u_id, v_id = u, v
                valid_pair = True
                break
        
        if not valid_pair:
            break
            
        merge_round += 1
        
        u_vec = graph.vectors[u_id]
        v_vec = graph.vectors[v_id]
        u_size = graph.sizes[u_id]
        v_size = graph.sizes[v_id]
        
        new_size = u_size + v_size
        new_vector = (u_vec * u_size + v_vec * v_size) / new_size
        new_node_id = next_node_id
        next_node_id += 1
        
        neighbors_u = graph.adj[u_id]
        neighbors_v = graph.adj[v_id]
        
        valid_new_neighbors = (neighbors_u | neighbors_v)
        valid_new_neighbors.discard(u_id)
        valid_new_neighbors.discard(v_id)
        valid_new_neighbors = {n for n in valid_new_neighbors if n in graph.active_nodes}
        
        graph.add_node(new_node_id, new_vector, new_size, valid_new_neighbors)
        merge_tree[new_node_id] = (u_id, v_id)
    
        
        p_new_ids = []
        q_new_ids = []
        
        for neighbor in valid_new_neighbors:
            graph.adj[neighbor].add(new_node_id)
            graph.adj[neighbor].discard(u_id)
            graph.adj[neighbor].discard(v_id)
            graph.update_degree(neighbor, len(graph.adj[neighbor]))
            
            p_new_ids.append(new_node_id)
            q_new_ids.append(neighbor)
            
        graph.remove_node(u_id)
        graph.remove_node(v_id)
        
        if p_new_ids:
            new_losses = calculate_lm2_loss_fast(
                p_new_ids, q_new_ids, graph
            )
            
            for loss, q_id in zip(new_losses, q_new_ids):
                low, high = (new_node_id, q_id) if new_node_id < q_id else (q_id, new_node_id)
                heapq.heappush(heap, (loss, low, high))

        if verbose and merge_round % 100 == 0:
            log_status()

    final_time = time.time() - start_time
    print(f'final_time: {final_time:.2f}s')
    data_to_save = {
        "merge_tree": merge_tree,
        "final_node_ids": list(graph.active_nodes)
    }

    
    return data_to_save