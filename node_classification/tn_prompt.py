import torch
import numpy as np
import os
import argparse
import sys


def main():

    parser = argparse.ArgumentParser(description="test node text prompt")
    
    parser.add_argument("--graph_name", type=str, required=True)
    parser.add_argument("--relax", type=str, required=True)
    parser.add_argument("--path_data", type=str, required=True)
    parser.add_argument("--ratio", type=str, required=True)

    parser.add_argument("--graph_type", type=str, default="")
    parser.add_argument("--node_type", type=str, default="")

    args = parser.parse_args()

    relax = args.relax
    graph_name = args.graph_name
    PATH_data = args.path_data
    ratio = args.ratio

    graph_type = args.graph_type
    node_type = args.node_type


    if relax == '1':
        infer_res = torch.load(f"./{graph_name}/inference_results_{ratio}_rel.pt")
    else:
        infer_res = torch.load(f"./{graph_name}/inference_results_{ratio}.pt")


    data = torch.load(f'{PATH_data}/dataset/{graph_name}.pt', weights_only=False)
    label_whol = data['label_whol']
    raw_texts = data['raw_texts']


    template = """
        Given a {GRAPH_TYPE} graph, the target {NODE_TYPE} has the following information:
        {TARGET_RAW_TEXT}
        The target {NODE_TYPE} is related to the following {NODE_TYPE} summary:
        {SUMMARY_TEXTS}
        Question: Based on the features of the target {NODE_TYPE} and the {NODE_TYPE} summary, please determine the most appropriate {GRAPH_TYPE} sub-category for the target {NODE_TYPE}.
        Categories: {CATEGORY_LIST}.
        Please think about the categorization of the target {NODE_TYPE} in a structured manner, and only output the single most relevant category of the target {NODE_TYPE}. Do not give any reasoning or extra text for your answer.
        Answer:
        """

    GRAPH_NAME = graph_name
    GRAPH_TYPE = graph_type
    CATEGORY_LIST = label_whol
    NODE_TYPE = node_type

    tn2prompt = {}
    for node_tuple, sum_text in infer_res.items():
        for node in node_tuple:
            TARGET_RAW_TEXT = raw_texts[node]
            SUMMARY_TEXTS = sum_text.replace("\n", "")
            final_prompt = template.format(
                                        NODE_TYPE=NODE_TYPE,
                                        GRAPH_TYPE=GRAPH_TYPE,
                                        CATEGORY_LIST=CATEGORY_LIST,
                                        TARGET_RAW_TEXT=TARGET_RAW_TEXT,
                                        SUMMARY_TEXTS=SUMMARY_TEXTS)
            tn2prompt[node] = final_prompt


    if relax == '1':
        torch.save(tn2prompt, f'./{graph_name}/tn2prompt_{ratio}_rel.pt')
    else:
        torch.save(tn2prompt, f'./{graph_name}/tn2prompt_{ratio}.pt')

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
