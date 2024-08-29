#!/usr/bin/env python
# coding: utf-8
import os
import json
import sys
import argparse
import torch
from datasets import load_dataset, Dataset
from transformers import pipeline, LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset

import numpy as np 
import random as rn

from inference import FullDialog, SlidingWindow, Summary, RAG, InfiniTransformer, InfiniTransformerBonus, ChainOfThought, GraphRAG
from utils import save, get_files, MODELS

import prompt as prompting
import os
import time 
os.environ['HF_HOME'] = '/home/tcharlot/models/'
os.environ["TOKENIZERS_PARALLELISM"] = "false"  

def init_seeds():
    SEED = 42
    rn.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference on multiple relations"
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="Directory of the dataset",
    )

    parser.add_argument(
        "----model_name_or_path",
        type=str,
        default=None,
        help="Repo of the model you want to use",
    )
    args = parser.parse_args()
    return args

def main():

    init_seeds()

    args = parse_args()

    hf_token = os.environ['HF_TOKEN']

    #TODO Jobs should be ran with different models through code args

    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    #model_id = "meta-llama/Meta-Llama-3-70B-Instruct"
    model_id = "google/gemma-2-27b-it"
    #model_id = "google/gemma-2-9b-it"
    
    #TODO remember to run only on A100 40 or 80
    runs = [FullDialog, # max / 2048 
            SlidingWindow, #2048 ?  
            Summary, #2048 
            RAG, #2048
            #InfiniTransformer, # 2048 * 4 = 8192 
            #InfiniTransformerBonus, 
            #ChainOfThought, 
            #GraphRAG,
            ]

    
    devices = []

    if torch.cuda.is_available() :
            nb_devices = torch.cuda.device_count()
            print("NB devices : ",nb_devices)
            for i in range(nb_devices):
                print("device ",i, " : ", torch.cuda.get_device_name(i))
                devices += [torch.cuda.get_device_name(i)]
    

    print(devices)
    if any(["8000" in d for d in devices]) or "gemma" in model_id or any(["V100" in d for d in devices]):
        attn_implementation = 'sdpa'
        print("using sdpa")
    else : 
        attn_implementation = 'flash_attention_2'
        print("using flash_attention_2")

    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                 attn_implementation=attn_implementation,
                                                 torch_dtype = torch.bfloat16,
                                                 device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    parameters = {"hf_token":hf_token,
                  "model_id":model_id,
                  "devices":devices,
                  "tokenizer":tokenizer,
                  "CoT":False}

    print("model id : ", model_id)
    infer_times = []
    directories = [ f.path for f in os.scandir("../data/meetup_target/") if f.is_dir() ]
    for d in directories:
        print("LAUNCHING : -------    ", d.split("/")[-1], "    -------")
        for run in runs:
            files = get_files(run, model_id, CoT=parameters["CoT"], dataset_name=d.split("/")[-1]) 
            if files == []:
                print("Already executed :", run.__name__.split(".")[-1] )
                continue
            #files = ["../data/meetup_target/Inferred/merge_309_387_26_37.csv"]
            print("Running : ", run.__name__.split(".")[-1])
            parameters["run"]=run
            parameters["dataset_name"]=d.split("/")[-1]
            if run.__name__.split(".")[-1] == "InfiniTransformer":
                model_name = "Infini-Llama3.1-8B-it-8192"
            else :
                model_name = MODELS[model_id]
            infer_times = d.split("/")[-1] + ","+ str(parameters["CoT"]) +","+ run.__name__.split(".")[-1] +","+ model_name +","+str(run.inference(model=model, files = files, **parameters))
            path_infer_stats = "../runs/meetup_target/inference_stats.csv"
            if os.path.exists(path_infer_stats):
                with open(path_infer_stats, "a") as f:
                    f.write("\n"+infer_times)
                    f.close()
            else :
                with open(path_infer_stats, "w") as f:
                    f.write("Category,Run,model,CoT,time\n"+infer_times)
                    f.close()
            

    print(infer_times)

if __name__ == '__main__':
    main()