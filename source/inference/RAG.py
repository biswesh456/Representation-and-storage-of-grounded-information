from .models import llama, gemma
from transformers import AutoTokenizer, AutoModel
import torch

def inference( files, **parameters):
    
    model_id = parameters['model_id']
    #modify max_length to have toggle sliding window or not
    parameters['max_length']=4096
    parameters['max_length']=2048
    parameters['processing']='rag'
    
    parameters['overlap']=2
    parameters['chunk_size']=5

    #parameters['overlap']=1
    #parameters['chunk_size']=3

    parameters["rag_tokenizer"] = AutoTokenizer.from_pretrained('nvidia/NV-Retriever-v1')
    parameters["rag_model"] = AutoModel.from_pretrained('nvidia/NV-Retriever-v1', trust_remote_code=True, torch_dtype=torch.float16)



    if 'llama' in model_id:
        res = llama.inference(files=files, verbose=True, **parameters)
    if "gemma" in model_id:
        res = gemma.inference(files=files, verbose=True, **parameters)

    return res