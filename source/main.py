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



from inference import FullDialog, SlidingWindow, Summary, RAG, InfiniTransformer, InfiniTransformerBonus, ChainOfThought, GraphRAG
from utils import save, get_files

import prompt as prompting
import os
import time 
os.environ['HF_HOME'] = '/home/tcharlot/models/'

def main():
    hf_token = os.environ['HF_TOKEN']

    #TODO Jobs should be ran with different models through code args

    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    #model_id = "meta-llama/Meta-Llama-3-70B-Instruct"
    #model_id = "google/gemma-2-27b-it"
    #model_id = "google/gemma-2-9b-it"
    
    #TODO remember to run only on A100 40 or 80
    runs = [FullDialog,
            SlidingWindow,
            Summary, 
            RAG, 
            InfiniTransformer, 
            InfiniTransformerBonus, 
            ChainOfThought, 
            GraphRAG]

    #prompts, answers = prompting.load_prompt("../../data/temporal") #TODO connect to real data
    
    #dataset = [[{"role":"user","content":prompt}] for prompt in prompts]
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
                  "tokenizer":tokenizer}

    print("model id : ", model_id)
    
    for run in runs: 
        #TODO Modify model if not traditional one
        files = get_files(run, model_id) #TODO change to dataset instead of files
        if files == []:
            print("Already executed :", run.__name__.split(".")[-1] )
            continue
        print("Running : ", run.__name__.split(".")[-1])
        parameters["run"]=run
        run.inference(model=model, files = files, **parameters) #TODO add dynamic saves with no impact on inference
        
if __name__ == '__main__':
    main()