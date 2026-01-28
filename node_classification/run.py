import subprocess
import os


GLOBAL_CONFIG = {
    "relax": '1',
    "graph_name": "citeseer", # book, dblp
    "path_data": '../dataset',
    "ratio": 0.5
}

if GLOBAL_CONFIG["graph_name"] == "ogb_arxiv" or GLOBAL_CONFIG["graph_name"] == "dblp" or GLOBAL_CONFIG["graph_name"] == "citeseer":
    graph_type = 'citation'
    node_type = "paper"
    info_i = "title"
    info_ii = "abstract"


if GLOBAL_CONFIG["graph_name"] == "book":
    graph_type = "book similarity"
    node_type = "book"
    info_i = "title"
    info_ii = "description"


if GLOBAL_CONFIG["graph_name"] == "products":
    graph_type = "co-purchase"
    node_type = "product"
    info_i = "name"
    info_ii = "description"


#Below are your environment paths, where `path_script` is the runtime environment and `path_interpreter` is the LLM inference environment.“
path_script = 'please input your env path'
path_interpreter = 'please input your env path'


SCRIPTS_TO_RUN = [
    {
        "script_path": f"{path_script}/sp_text.py",
        "interpreter": f"{path_interpreter}/gs/bin/python",
        "script_args": {
            "graph_type": graph_type,
            "node_type": node_type,
            "info_i": info_i,
            "info_ii": info_ii,
        }
    },
    {
        "script_path": f"{path_script}/vllm_infer.py",
        "interpreter": f"{path_interpreter}/vllm/bin/python"
    },
    {
        "script_path": f"{path_script}/tn_prompt.py",
        "interpreter": f"{path_interpreter}/gs/bin/python", 
        "script_args": {
            "graph_type": graph_type,
            "node_type": node_type
        }
    },
    {
        "script_path": f"{path_script}/node_classification.py",
        "interpreter": f"{path_interpreter}/vllm/bin/python"
    }
]


def run_script_with_env(script_info):

    script = script_info["script_path"]
    interpreter = script_info["interpreter"]
    script_specific_args = script_info.get("script_args", {})

    all_args = {**GLOBAL_CONFIG, **script_specific_args}

    arg_list = []
    for key, value in all_args.items():
        arg_list.append(f"--{key.lower()}")
        arg_list.append(str(value))
        

    command = [interpreter, script] + arg_list
    
    print(f"--- running script '{script}' ---")
    print(f"full command: {' '.join(command)}")

    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        
        print(f"script '{script}' running successfully")
        print(result.stdout)
        
    except subprocess.CalledProcessError as e:
        print(f"script '{script}' running unsuccessfully, exit code: {e.returncode}")
        print("--- error (STDERR) ---")
        print(e.stderr)
    except FileNotFoundError:
        print(f"error: script '{script}' unsuccessfully. cant find interpreter {interpreter}")
    
    print("-" * 50)

if __name__ == "__main__":
    for script_data in SCRIPTS_TO_RUN:
        run_script_with_env(script_data)