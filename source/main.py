#!/usr/bin/env python
# coding: utf-8
import os
import json
import sys
import argparse
import torch
from datasets import load_dataset, Dataset
from transformers import pipeline, LlamaForCausalLM, LlamaTokenizer, AutoTokenizer
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset



from inference import FullDialog, SlidingWindow, Summary, RAG, InfiniTransformer, InfiniTransformerBonus, ChainOfThought, GraphRAG


import prompt as prompting
import os
import time 
os.environ['HF_HOME'] = '/home/tcharlot/models/'

def main():
    hf_token = os.environ['HF_TOKEN']

    #TODO Jobs should be ran with different models through code args
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    #model_id = "google/gemma-2-27b-it"
    #model_id = "google/gemma-2-9b-it"
    

    runs = [FullDialog,
            SlidingWindow,
            Summary, 
            RAG, 
            InfiniTransformer, 
            InfiniTransformerBonus, 
            ChainOfThought, 
            GraphRAG]

    prompts, answers = prompting.load_prompt("../../data/temporal") #TODO connect to real data
    
    dataset = [[{"role":"user","content":prompt}] for prompt in prompts]
    devices = []
    if torch.cuda.is_available() :
            nb_devices = torch.cuda.device_count()
            print("NB devices : ",nb_devices)
            for i in range(nb_devices):
                print("device ",i, " : ", torch.cuda.get_device_name(i))
                devices += [torch.cuda.get_device_name(i)]

    parameters = {"hf_token":hf_token,
                  "model_id":model_id,
                  "devices":devices}

    for run in runs: 
        #TODO add if case to check if already processed
        res = run.inference(dataset = dataset, **parameters)
        print(res)
if __name__ == '__main__':
    main()