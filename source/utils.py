import pandas as pd
from glob import glob
import os
from transformers import TextGenerationPipeline
from transformers.pipelines.text_generation import ReturnType, Chat


MODELS = {"meta-llama/Meta-Llama-3-8B-Instruct":"Llama3-8B-it",
              "meta-llama/Meta-Llama-3-70B-Instruct":"Llama3-70B-it",
              "google/gemma-2-27b-it":"Gemma-2-27B-it",
              "google/gemma-2-9b-it":"Gemma-2-9B-it",
              }

def save(data, run, file, model_id):

    directory = "../runs/" + run.__name__.split(".")[-1] +"/"
    if not os.path.exists(directory):
        os.mkdir(directory)
    directory += MODELS[model_id] + "/"
    if not os.path.exists(directory):
        os.mkdir(directory)
    with open(directory+file.split("/")[-1].split(".")[0]+".txt", "w") as file:
        file.write(data)
        file.close()

def get_files(run, model_id):
    #must return the files that have not been executed
    dataset = glob("../../data/temporal/*.csv") #TODO change to real dataset
    files = None
    directory = "../runs/" + run.__name__.split(".")[-1] +"/" 
    if os.path.exists(directory):
        directory += MODELS[model_id]
        existing_files = glob(directory+"/*.txt")
        temp_existing = set([f.split("/")[-1].split(".")[0] for f in existing_files])
        temp_dataset = set([f.split("/")[-1].split(".")[0] for f in dataset])
        files = list(temp_dataset - temp_existing)
        #print(files)
        files = ["../../data/temporal/"+f+".csv" for f in files]#TODO change to real dataset
    else: 
        print("Not implemented :", run.__name__.split(".")[-1]  )
        files = dataset
    return files