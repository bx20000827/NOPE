import torch
import random
import numpy as np
from sentence_transformers import SentenceTransformer

import argparse




label_whol_dict = {
    'ogb_arxiv': {
        0: "Numerical Analysis",
        1: "Multimedia",
        2: "Logic in Computer Science",
        3: "Computers and Society",
        4: "Cryptography and Security",
        5: "Distributed, Parallel, and Cluster Computing",
        6: "Human-Computer Interaction",
        7: "Computational Engineering, Finance, and Science",
        8: "Networking and Internet Architecture",
        9: "Computational Complexity",
        10: "Artificial Intelligence",
        11: "Multiagent Systems",
        12: "General Literature",
        13: "Neural and Evolutionary Computing",
        14: "Symbolic Computation",
        15: "Hardware Architecture",
        16: "Computer Vision and Pattern Recognition",
        17: "Graphics",
        18: "Emerging Technologies",
        19: "Systems and Control",
        20: "Computational Geometry",
        21: "Other Computer Science",
        22: "Programming Languages",
        23: "Software Engineering",
        24: "Machine Learning",
        25: "Sound",
        26: "Social and Information Networks",
        27: "Robotics",
        28: "Information Theory",
        29: "Performance",
        30: "Computation and Language",
        31: "Information Retrieval",
        32: "Mathematical Software",
        33: "Formal Languages and Automata Theory",
        34: "Data Structures and Algorithms",
        35: "Operating Systems",
        36: "Computer Science and Game Theory",
        37: "Databases",
        38: "Digital Libraries",
        39: "Discrete Mathematics"
        },
    'book': {
        0: 'Children',
        1: 'Comics, Graphic',
        2: 'Fantasy, Paranormal',
        3: 'History, Historical fiction, Biography',
        4: 'Mystery, Thriller, Crime',
        5: 'Poetry',
        6: 'Romance',
        7: 'Young-adult'
        },
    'citeseer': {
        0: "Agents",
        1: "Machine Learning",
        2: "Information Retrieval",
        3: "Databases",
        4: "Human-Computer Interaction",
        5: "Artificial Intelligence"
        },
    'products': {
        0: "Home & Kitchen",
        1: "Health & Personal Care",
        2: "Beauty",
        3: "Sports & Outdoors",
        4: "Books",
        5: "Patio, Lawn & Garden",
        6: "Toys & Games",
        7: "CDs & Vinyl",
        8: "Cell Phones & Accessories",
        9: "Grocery & Gourmet Food",
        10: "Arts, Crafts & Sewing",
        11: "Clothing, Shoes & Jewelry",
        12: "Electronics",
        13: "Movies & TV",
        14: "Software",
        15: "Video Games",
        16: "Automotive",
        17: "Pet Supplies",
        18: "Office Products",
        19: "Industrial & Scientific",
        20: "Musical Instruments",
        21: "Tools & Home Improvement",
        23: "Baby Products",
        24: "label 25",
        25: "Appliances",
        26: "Kitchen & Dining",
        27: "Collectibles & Fine Art",
        28: "All Beauty",
        30: "Amazon Fashion",
        31: "Computers",
        32: "All Electronics",
        34: "MP3 Players & Accessories",
        36: "Office & School Supplies",
        37: "Home Improvement",
        38: "Camera & Photo",
        39: "GPS & Navigation",
        40: "Digital Music",
        41: "Car Electronics",
        42: "Baby",
    }

}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="citeseer")
    # data_cons
    parser.add_argument("--feat_flag", type=bool, default=True)
    parser.add_argument("--datasplit_flag", type=bool, default=True)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    # seed
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()

args = parse_args()



# random seed setting
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)



class graph_dataset():
    def __init__(self, args):
        super(graph_dataset, self).__init__()
        self.args = args
        self.graph_name = self.args.dataset
        self.data = self._load_data()

        if self.args.feat_flag == True:
            self._feat_cons()
        if self.args.datasplit_flag == True:
            self._datasplit()
        if self.args.feat_flag == True or self.args.datasplit_flag == True:
            torch.save(self.data, f'./{self.graph_name}.pt')


        if self.args.dataset in ['ogb_arxiv', 'citeseer']:
            self.dataset_name_in_prompt = "Ogb_ArXiv"
            self.graph_type = "academic citation"
            self.node_type = "paper"
        elif self.args.dataset in ['book']:
            self.dataset_name_in_prompt = "Book"
            self.graph_type = "book similarity"
            self.node_type = "book"
        elif self.args.dataset in ['products']:
            self.dataset_name_in_prompt = "Products"
            self.graph_type = "product-purchased"
            self.node_type = "product"


        else:
            raise NotImplementedError
    
    def _load_data(self):
        return torch.load(f'./{self.graph_name}.pt', weights_only=False)

    def _feature_normalize(self, X, eps=1e-10):

        if not isinstance(X, torch.Tensor):
            raise TypeError("X must be a PyTorch tensor")
        
        mean = X.mean(dim=0, keepdim=True)
        X_centered = X - mean
        
        norms = torch.norm(X_centered, p=2, dim=1, keepdim=True)
        norms = torch.maximum(norms, torch.tensor(eps, device=X.device, dtype=X.dtype))
        
        X_final = X_centered / norms
        
        return X_final


    def _feat_cons(self):
        model = SentenceTransformer('./bert_v2') # your own bert path
        embeddings = model.encode(self.data['raw_texts'], 
                                batch_size=32, 
                                show_progress_bar=True)
        embeddings = torch.tensor(embeddings)
        embeddings = self._feature_normalize(embeddings)
        self.data['feat'] = embeddings


    def _datasplit(self):
        num_nodes = self.data.x.shape[0]
        num_test = round(num_nodes * self.args.test_ratio)
        perm = torch.randperm(num_nodes)
        test_indices = perm[:num_test]
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask[test_indices] = True
        self.data['test_mask'] = test_mask
        self.data['num_nodes'] = num_nodes


graph_ = graph_dataset(args=args)