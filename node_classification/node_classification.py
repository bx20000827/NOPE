import torch
import os
from vllm import LLM, SamplingParams
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="test node classification")
    parser.add_argument("--graph_name", type=str, required=True)
    parser.add_argument("--relax", type=str, required=True)
    parser.add_argument("--path_data", type=str, required=True)
    parser.add_argument("--ratio", type=str, required=True)


    args = parser.parse_args()

    relax = args.relax
    graph_name = args.graph_name
    ratio = args.ratio

    if relax == '1':
        tn2prompt = torch.load(f'./{graph_name}/tn2prompt_{ratio}_rel.pt', weights_only=False)
    else:
        tn2prompt = torch.load(f'./{graph_name}/tn2prompt_{ratio}.pt', weights_only=False) 


    llm = LLM(
        model="./Llama-3-8B-Instruct", # Replace with the address of your own LLM model
        tensor_parallel_size=1, 
        dtype="float16"
    )

    sampling_params = SamplingParams(
        temperature=0, 
        max_tokens=64
    )

    prompts = []
    metadata = [] 

    for node_id, final_prompt in tn2prompt.items():
        prompts.append(final_prompt)
        metadata.append(node_id)

    print(f"Starting inference on {len(prompts)} prompts...")
    outputs = llm.generate(prompts, sampling_params)

    id2res = {}


    for i, output in enumerate(outputs):
        node_id = metadata[i]
        generated_text = output.outputs[0].text
        
        id2res[node_id] = generated_text

    print("Inference finished.")

    if relax == '1':
        torch.save(id2res, f"./{graph_name}/tn_class_results_{ratio}_rel.pt")
    else:
        torch.save(id2res, f"./{graph_name}/tn_class_results_{ratio}.pt")

    del llm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()