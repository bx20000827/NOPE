import torch
from model import merge_relax, merge
import utils
import pickle
from ogb.nodeproppred import PygNodePropPredDataset, NodePropPredDataset


# dataload
relax = False
merge_ratio = 0.3
graph_name = 'citeseer'

# log
PATH = './logger'
LOG_DIR = f'{PATH}/logs'
logger = utils.build_logger(LOG_DIR)

logger.info(f"start loading: {graph_name}")


PATH_data = './dataset'
data = torch.load(f'{PATH_data}/dataset/{graph_name}.pt', weights_only=False)
# merge
embeddings = data['feat']
edge_index = data['edge_index']
n_nodes = data['num_nodes']


if relax:
    info = merge_relax.hierarchical_merge_rel(embeddings, 
                                            edge_index,
                                            n_nodes, 
                                            merge_ratio = merge_ratio,
                                            verbose = True,
                                            logger = logger)

    
    with open(f'./new_info_{merge_ratio}_rel.pkl', 'wb') as f:
        pickle.dump(info, f)


else:
    info = merge.hierarchical_merge(embeddings, 
                                    edge_index,
                                    n_nodes, 
                                    merge_ratio = merge_ratio,
                                    verbose = True,
                                    logger = logger)


    with open(f'./new_info_{merge_ratio}.pkl', 'wb') as f:
        pickle.dump(info, f)
