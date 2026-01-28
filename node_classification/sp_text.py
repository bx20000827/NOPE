import torch
from torch.nn import functional as F
from tqdm import tqdm
from transformers import AutoTokenizer
import argparse
import sys
import pickle

import numpy as np

def reconstruct_and_average(
    data_saved: dict, 
    original_embeddings: np.ndarray
) -> dict:

    merge_tree = data_saved["merge_tree"]
    final_node_ids = data_saved["final_node_ids"]
    
    mapping = {}
    super_vectors = {}
    
    print(f"{len(final_node_ids)} supernodes...")
    
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
    
    print("finish...")
    

    for root_id, leaf_indices in mapping.items():
 
        mean_vector = np.mean(original_embeddings[leaf_indices], axis=0)
        
        super_vectors[root_id] = mean_vector.astype(np.float32)

    print("finish computing。")
    
    return {
        "mapping": mapping,
        "super_vectors": super_vectors
    }

class classi_prompt():
    def __init__(self, LLM):
        self.TEMPLATE = """
            Given a group of {NODE_TYPE}s from the same community within a {GRAPH_TYPE} graph. Each {NODE_TYPE} is represented by its {INFO_I} and {INFO_II}.
            {NODE_TYPE}s: {RAW_TEXT}
            Task: Summarize the content of these {NODE_TYPE}s that are semantically aligned with each other. The summary must explicitly articulate the intersection of their work.
            Format requirement: (1) Identify the major recurring themes/directions. (2) Structure the summary around these identified themes. (3) Write the final summary in a cohesive and formal academic style.
            Answer:
            """

        self.LLM = LLM

        if self.LLM == 'llama':
            self.model_dir = "./Llama-3-8B-Instruct" # Replace with the address of your own LLM model
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, trust_remote_code=True)
        self.max_token = 5000 

    def _count_tokens(self, text):
        return len(self.tokenizer(text)["input_ids"])

    def build(self, GRAPH_TYPE, NODE_TYPE, NEIG_TEXT, INFO_I, INFO_II):
        base_prompt = self.TEMPLATE.format(
            GRAPH_TYPE = GRAPH_TYPE,
            NODE_TYPE = NODE_TYPE,
            INFO_I = INFO_I,
            INFO_II = INFO_II,
            RAW_TEXT = "", 
        )

        base_tokens = self._count_tokens(base_prompt)
        max_input = self.max_token - 198
        
        filled_texts = []
        cur_len = base_tokens
        count = 1
        for nei in NEIG_TEXT:
            t = self._count_tokens(nei)
            if cur_len + t > max_input:
                break
            filled_texts.append(f"{count}. {nei}")
            count += 1
            cur_len += t

        final_prompt = self.TEMPLATE.format(
            GRAPH_TYPE=GRAPH_TYPE,
            NODE_TYPE=NODE_TYPE,
            INFO_I = INFO_I,
            INFO_II = INFO_II,
            RAW_TEXT="\n".join(filled_texts),
        )

        return final_prompt

def group_nodes_by_supernode(nodes, node2sp, sp2nodelist):

    sp2input_nodes = {}
    for node in nodes:
        sp = node2sp.get(node)
        if sp is None:
            continue  
        if sp not in sp2input_nodes:
            sp2input_nodes[sp] = []
        sp2input_nodes[sp].append(node)
    
    result = {}
    for sp, input_node_list in sp2input_nodes.items():
        input_node_tuple = tuple(input_node_list)
        result[input_node_tuple] = sp2nodelist[sp]
    
    return result



def main():
    parser = argparse.ArgumentParser(description="super node text prompt")
    
    parser.add_argument("--graph_name", type=str, required=True)
    parser.add_argument("--relax", type=str, required=True)
    parser.add_argument("--path_data", type=str, required=True)
    parser.add_argument("--ratio", type=float, required=True)

    parser.add_argument("--graph_type", type=str, default="")
    parser.add_argument("--node_type", type=str, default="")
    parser.add_argument("--info_i", type=str, default="")
    parser.add_argument("--info_ii", type=str, default="")

    args = parser.parse_args()

    relax = args.relax
    graph_name = args.graph_name
    PATH_data = args.path_data
    ratio = args.ratio

    graph_type = args.graph_type
    node_type = args.node_type
    info_i = args.info_i
    info_ii = args.info_ii


    # device = 'cpu'
    data = torch.load(f'{PATH_data}/dataset/{graph_name}.pt', weights_only=False)
    raw_texts = data['raw_texts']
    nodes = data['test_mask'].nonzero().squeeze().tolist()
    x = data['feat']
    embeddings = x.numpy()


    if relax == '1':
        with open(f'../new_info_{ratio}_rel.pkl', 'rb') as f:
            data_saved = pickle.load(f)
    else:
        with open(f'../new_info_{ratio}.pkl', 'rb') as f:
            data_saved = pickle.load(f)

    info_all = reconstruct_and_average(data_saved, embeddings)
    sp2nodelist = info_all['mapping']


    node2sp = dict()
    for sp, node_list in sp2nodelist.items():
        for node in node_list:
            node2sp[node] = sp
    
    result = group_nodes_by_supernode(nodes, node2sp, sp2nodelist)

    filter_res = dict()
    for query_tuple, node_list in result.items():
        query_set = set(query_tuple)
        nnl = [i for i in node_list if i not in query_set]
        filter_res[query_tuple] = nnl


    id2sortid = dict()
    for id_tuple, nodeset in filter_res.items():
        node_tuple = torch.tensor(list(id_tuple), dtype=torch.long)
        ave_emb = torch.mean(x[node_tuple], dim=0)
        ave_emb = F.normalize(ave_emb, p=2, dim=0)

        candidate_nodes = torch.tensor(list(nodeset), dtype=torch.long)
        candidate_emb = x[candidate_nodes]

        cos_sim = F.cosine_similarity(ave_emb.unsqueeze(0), candidate_emb, dim=1)
        sorted_indices = torch.argsort(cos_sim, descending=True)
        sorted_nodes = candidate_nodes[sorted_indices]
        sorted_nodes = sorted_nodes.cpu().tolist()

        to_remove = set(id_tuple)
        new_sorted_nodes = [x for x in sorted_nodes if x not in to_remove]
        
        id2sortid[id_tuple] = new_sorted_nodes[:10]


    max_len = 10

    id2text = dict()
    for id_tuple, sorted_ids in tqdm(id2sortid.items()):
        id_len = len(id_tuple)
        candi_text = [raw_texts[idx] for idx in id_tuple]
        if max_len > id_len:
            candi = sorted_ids[:max_len - id_len]
            candi_text.extend([raw_texts[idx] for idx in candi])
        else:
            candi_text = candi_text[:max_len]
        id2text[id_tuple] = candi_text


    prompt = classi_prompt(LLM='llama')
    id2prompt = {}

    for node_tuple, text in id2text.items():
        id2prompt[node_tuple] = prompt.build(graph_type, node_type, text, info_i, info_ii)

    if relax == '1':
        torch.save(id2prompt, f'./{graph_name}/id2prompt_{ratio}_rel.pt')
    else:
        torch.save(id2prompt, f'./{graph_name}/id2prompt_{ratio}.pt')
    

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':

    main()

  