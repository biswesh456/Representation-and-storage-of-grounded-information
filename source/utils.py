import pandas as pd
from glob import glob
import os
from transformers import TextGenerationPipeline
from transformers.pipelines.text_generation import ReturnType, Chat
import json

MODELS = {"meta-llama/Meta-Llama-3-8B-Instruct":"Llama3-8B-it",
          "meta-llama/Meta-Llama-3.1-8B-Instruct":"Llama3.1-8B-it",
          "meta-llama/Meta-Llama-3.1-70B-Instruct":"Llama3.1-70B-it",
          "meta-llama/Meta-Llama-3-70B-Instruct":"Llama3-70B-it",
          "google/gemma-2-27b-it":"Gemma-2-27B-it",
          "google/gemma-2-9b-it":"Gemma-2-9B-it",
          "llama-3.1-8b-infini-noclm-8192": "Infini-Llama3.1-8B-it-8192",
          "llama-3.1-8b-infini-noclm-gated": "Infini-Llama3.1-8B-it-gated",
          "google/gemma-2-2b-it":"Gemma-2-2B-it",
          "mistralai/Mistral-Nemo-Instruct-2407": "Mistral-Nemo-it",
          "Qwen/QwQ-32B":"Qwen-QwQ-32B"
        }

def save(data, run, file, model_id, optional_arg=None, CoT=False, dataset_name=None):
    if CoT:
        CoT_str = "CoT"
    else :
        CoT_str = ""
    directory = "../runs/"+dataset_name+"/"
    print("DATASET",dataset_name)
    directory += run.__name__.split(".")[-1]+ CoT_str +"/"
    if optional_arg is not None:
        directory = "../data/"+optional_arg+"/"+dataset_name+"/"
    if not os.path.exists(directory):
        print("DIRECTORY : ", directory)
        os.makedirs(directory, exist_ok=True)#TODO mkdir not working
    if model_id != "" : 
        directory += MODELS[model_id] + "/"
    if not os.path.exists(directory):
        os.mkdir(directory)

    df = pd.read_csv(file)
    if "spot" in dataset_name:
        answer = df.iloc[-3]["utterance"]
        exact = df.iloc[-2]["utterance"]
        question = df.iloc[-4]["utterance"]
        json_dict = {"question":question,
                     "completion":data,
                     "exact_match":exact,
                     "label":answer,}
    else :
        idx = file.split("/")[-1]
        question = df.iloc[-2]["msg"]
        with open("../data/answers/"+run.__name__.split(".")[-1]+"/"+idx.split(".")[0]+".txt", "r") as f:
            lines = f.readlines()
            label = lines[0][:-1]
            exact_label = lines[1]
            f.close()    
        json_dict = {"question":question,
                     "completion":data,
                     "exact_match":exact_label,
                     "label":label,}
    with open(directory+file.split("/")[-1].split(".")[0]+".json", "w") as file:
        json.dump(json_dict,file, indent=4)
        file.close()
        

def get_files(run, model_id, optional_arg=None, CoT=False, dataset_name=None):
    #must return the files that have not been executed
    if run == "InfiniTransformer" : 
        model_id = "Infini-Llama3.1-8B-it-8192"


    dataset = glob("../data/"+dataset_name+"/*.csv")

    files = None
    if CoT:
        CoT_str = "CoT"
    else :
        CoT_str = ""
    directory = "../runs/"+ dataset_name +"/"+ run.__name__.split(".")[-1]+ CoT_str +"/" 
    if optional_arg is not None:
        directory =  "../data/"+optional_arg+"/"+dataset_name+"/"
    if os.path.exists(directory):
        if model_id != "" : 
            directory += MODELS[model_id]
        existing_files = glob(directory+"/*.json")
        temp_existing = set([f.split("/")[-1].split(".")[0] for f in existing_files])
        temp_dataset = set([f.split("/")[-1].split(".")[0] for f in dataset])
        files = list(temp_dataset - temp_existing)
        #print(files)
        files = ["../data/"+dataset_name+"/"+f+".csv" for f in files]
    else: 
        print("Not implemented :", run.__name__.split(".")[-1]  )
        files = dataset
    return files