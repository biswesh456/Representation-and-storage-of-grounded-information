#!/usr/bin/env python
# coding: utf-8
import os
os.environ['HF_HOME'] = '/data/almanach/user/cdearauj/models/'
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

from inference import FullDialog, SlidingWindow, Summary, RAG, RAGBM25, InfiniTransformer, InfiniTransformerBonus, ChainOfThought, GraphRAG, LocalAnswerContext
from utils import save, get_files, MODELS

import prompt as prompting
import os
import time 
import torch._dynamo
    

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
        "--model",
        type=str,
        default=None,
        help="Repo of the model you want to use",
    )
    parser.add_argument(
        "--cot",
        action='store_true',
        help="With or without Chain of Thought",
    )
    args = parser.parse_args()
    return args

def main():
    init_seeds()

    args = parse_args()

    hf_token = os.environ['HF_TOKEN']

    #TODO Jobs should be ran with different models through code args
    if args.model is None :
        #model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        #model_id = "meta-llama/Meta-Llama-3-70B-Instruct"
        #model_id = "google/gemma-2-27b-it"
        #model_id = "google/gemma-2-9b-it"
        model_id = "google/gemma-2-2b-it" # not implemented
        #model_id = "Qwen/QwQ-32B"
    else : 
        model_id = args.model


    
    print(args.cot)
    
    #TODO spot the difference inference and evaluation
    #TODO get results for memgpt
    #TODO add ifs for context size


    runs = [
            #FullDialog, # max / 2048 
            #SlidingWindow, #2048 ?  
            #Summary, #2048 
            RAG, #2048
            RAGBM25,
            #LocalAnswerContext
            #InfiniTransformer, # 2048 * 4 = 8192 
            #InfiniTransformerBonus,
            ]

    
    dataset = "merged_spot"
    #dataset = "meetup_target"

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
                                                 attn_implementation='flash_attention_2',
                                                 torch_dtype = torch.bfloat16,
                                                 device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    parameters = {"hf_token":hf_token,
                  "model_id":model_id,
                  "devices":devices,
                  "tokenizer":tokenizer,
                  "CoT":args.cot}

    if "spot" in dataset:
        parameters["max_new_tokens"] = 300
    print("model id : ", model_id)
    infer_times = []
    
    directories = [ f.path for f in os.scandir("../data/"+dataset+"/") if f.is_dir() ]
    for d in directories:
        print("LAUNCHING : -------    ", d.split("/")[-1], "    -------")
        for run in runs:
            files = get_files(run, model_id, CoT=parameters["CoT"], dataset_name=dataset+"/"+d.split("/")[-1]) 
            if files == []:
                print("Already executed :", run.__name__.split(".")[-1] )
                continue
            #files = ["../data/meetup_target/Inferred/merge_309_387_26_37.csv"]
            print("Running : ", run.__name__.split(".")[-1])
            parameters["run"]=run
            parameters["dataset_name"]=dataset+"/"+d.split("/")[-1]
            if run.__name__.split(".")[-1] == "InfiniTransformer":
                model_name = "Infini-Llama3.1-8B-it-8192"
            else :
                model_name = MODELS[model_id]
            #execution here
            infer_times = dataset+"/"+d.split("/")[-1] + ","+ str(parameters["CoT"]) +","+ run.__name__.split(".")[-1] +","+ model_name +","+str(run.inference(model=model, files = files, **parameters))
            path_infer_stats = "../runs/"+dataset+"/inference_stats.csv"
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
    os.environ['HF_HOME'] = '/data/almanach/user/cdearauj/theo/models/'
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  
    init_seeds()
    main()