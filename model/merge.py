import heapq
import numpy as np
import time
from typing import Dict, List, Tuple, Set, Optional
import torch

class GraphDataManagerM1:

    def __init__(self, initial_embeddings: np.ndarray, initial_sizes: np.ndarray, max_nodes: int):
        self.dim = initial_embeddings.shape[1]
        
    
        self.vectors = np.zeros((max_nodes, self.dim), dtype=np.float32)
        self.sizes = np.zeros(max_nodes, dtype=np.float32)
        self.sum_sq = np.zeros(max_nodes, dtype=np.float32)
        
        n = initial_embeddings.shape[0]
        self.vectors[:n] = initial_embeddings
        self.sizes[:n] = initial_sizes
        
        self.adj: Dict[int, Set[int]] = {}
        self.active_nodes = set(range(n))

    def add_node(self, new_id: int, vector: np.ndarray, size: float, neighbors: Set[int]):
        self.vectors[new_id] = vector
        self.sizes[new_id] = size
        self.adj[new_id] = neighbors
        self.active_nodes.add(new_id)

    def remove_node_logic(self, node_id: int):
        if node_id in self.adj:
            del self.adj[node_id]
        self.active_nodes.discard(node_id)
 


def calculate_loss_m1_fast(
    p_id: int,
    q_id: int,
    graph: GraphDataManagerM1
) -> float:

    s_p = graph.sizes[p_id]
    s_q = graph.sizes[q_id]
    weight = (s_p * s_q) / (s_p + s_q + 1e-8)
    
    if weight == 0: return 0.0

    sum_sq_p = graph.sum_sq[p_id]
    sum_sq_q = graph.sum_sq[q_id]
    

    
    neighbors_p = graph.adj.get(p_id, set())
    neighbors_q = graph.adj.get(q_id, set())
    
    if len(neighbors_p) < len(neighbors_q):
        common_neighbors = list(neighbors_p.intersection(neighbors_q))
    else:
        common_neighbors = list(neighbors_q.intersection(neighbors_p))
    
    sum_diff_sq = 0.0
    
    if common_neighbors:

        common_indices = np.array(common_neighbors, dtype=np.int32)
        k_vecs = graph.vectors[common_indices] 
        
        u_vec = graph.vectors[p_id]
        v_vec = graph.vectors[q_id]
        
 
        dots_u = k_vecs @ u_vec 
        dots_v = k_vecs @ v_vec
      
        diffs = dots_u - dots_v
        sum_diff_sq = np.sum(diffs ** 2)

    return weight * (sum_sq_p + sum_sq_q + sum_diff_sq)


def initialize_sum_sq(n_nodes: int, graph: GraphDataManagerM1):

    for i in range(n_nodes):
        neighbors = list(graph.adj.get(i, set()))
        if not neighbors:
            graph.sum_sq[i] = 0.0
            continue
            
        neighbor_indices = np.array(neighbors, dtype=np.int32)
        neighbor_vecs = graph.vectors[neighbor_indices]
        u_vec = graph.vectors[i]
        
        dots = neighbor_vecs @ u_vec
        graph.sum_sq[i] = np.sum(dots ** 2)


def initialize_graph_edges(edge_index: np.ndarray, num_nodes: int, graph: GraphDataManagerM1) -> Set[Tuple[int, int]]:
    rows, cols = edge_index
    all_edges = set()
    
    temp_adj = {i: set() for i in range(num_nodes)}
    for u, v in zip(rows, cols):
        u, v = int(u), int(v)
        if u == v: continue
        temp_adj[u].add(v)
        temp_adj[v].add(u)
        all_edges.add(tuple(sorted((u, v))))
    
    graph.adj = temp_adj
    return all_edges


def hierarchical_merge(
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

    max_possible_nodes = n_nodes * 2 + 100
    initial_sizes = np.ones(n_nodes, dtype=np.float32)
    
    graph = GraphDataManagerM1(embeddings, initial_sizes, max_possible_nodes)
    
    if verbose and logger:
        logger.info(f"GraphManager initialized. Matrix shape: {graph.vectors.shape}")

    processed_pairs = initialize_graph_edges(edge_index, n_nodes, graph)
    
    if verbose and logger:
        logger.info("Initializing sum_sq cache (vectorized)...")
    initialize_sum_sq(n_nodes, graph)

    heap = []
    
    if verbose and logger:
        logger.info("Initializing heap...")
        
    for u, v in processed_pairs:
        loss = calculate_loss_m1_fast(u, v, graph)
        heapq.heappush(heap, (loss, u, v))
    
    merge_tree: Dict[int, Tuple[int, int]] = {}
    merge_round = 0
    next_node_id = n_nodes
    
    def log_status(curr_loss):
        if logger:
            logger.info(f"Round {merge_round}: Active={len(graph.active_nodes)}, "
                        f"Loss={curr_loss:.4f}")

    if verbose and logger:
        logger.info(f"Start merging: {n_nodes} -> {target}")

    while len(graph.active_nodes) > target:
        
        valid_pair = False
        loss_val = 0.0
        u_id, v_id = -1, -1
        
        while heap:
            loss_val, u, v = heapq.heappop(heap)
            if (u in graph.active_nodes and v in graph.active_nodes and 
                v in graph.adj.get(u, set())):
                u_id, v_id = u, v
                valid_pair = True
                break
        
        if not valid_pair:
            if verbose and logger: logger.info("No valid pairs left.")
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

        if valid_new_neighbors:
            neighbor_list = list(valid_new_neighbors)
            n_indices = np.array(neighbor_list, dtype=np.int32)
            n_vecs = graph.vectors[n_indices]
            

            dots_new = n_vecs @ new_vector 
            
            graph.sum_sq[new_node_id] = np.sum(dots_new ** 2)
            

            dots_u_recalc = n_vecs @ u_vec
            dots_v_recalc = n_vecs @ v_vec
            
    

            is_neighbor_u = np.array([n in neighbors_u for n in neighbor_list], dtype=bool)
            is_neighbor_v = np.array([n in neighbors_v for n in neighbor_list], dtype=bool)
            
            updates = dots_new ** 2
       
            updates[is_neighbor_u] -= (dots_u_recalc[is_neighbor_u] ** 2)
            updates[is_neighbor_v] -= (dots_v_recalc[is_neighbor_v] ** 2)
            
            graph.sum_sq[n_indices] += updates
            np.maximum(graph.sum_sq[n_indices], 0, out=graph.sum_sq[n_indices])
            
            for n_id in neighbor_list:
                graph.adj[n_id].add(new_node_id)
                graph.adj[n_id].discard(u_id)
                graph.adj[n_id].discard(v_id)
 
                loss = calculate_loss_m1_fast(new_node_id, n_id, graph)
                low, high = (new_node_id, n_id) if new_node_id < n_id else (n_id, new_node_id)
                heapq.heappush(heap, (loss, low, high))
        else:
            graph.sum_sq[new_node_id] = 0.0

        graph.remove_node_logic(u_id)
        graph.remove_node_logic(v_id)
        
        if verbose and merge_round % 100 == 0:
            log_status(loss_val)

    if verbose and logger:
        total_time = time.time() - start_time
        logger.info(f"Complete. Time: {total_time:.2f}s")

    return {
        "merge_tree": merge_tree,
        "final_node_ids": list(graph.active_nodes)
    }