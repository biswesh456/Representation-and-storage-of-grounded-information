from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset
from transformers import pipeline
import torch
import time 
import prompt as prompting
from utils import save
from inference import FullDialog

def inference(files,
              devices,
              model_id,
              hf_token,
              max_length,
              run,
              model,
              tokenizer,
              verbose=False,
              processing=None,
              **kwargs):
    
    

    if any(["8000" in d for d in devices]):
        attn_implementation = 'sdpa'
    else : 
        attn_implementation = 'flash_attention_2'
        attn_implementation = 'sdpa'
        #attn_implementation = 'eager'
    

    print("Launching : ", model_id)

    pipe = pipeline("text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    token=hf_token,
                    max_length=max_length,
                    device_map="auto",
                    return_full_text =False,
                    #add_special_tokens=True,
                    )
    parameters ={"model_id":model_id,
                 "run":run,
                 "kwargs": kwargs}

    prompting.pre_generate(pipe, files, **parameters)

    prompts, answers = prompting.load_prompt(files, tokenizer=pipe.tokenizer, model_id=model_id, processing=processing) #TODO connect to real data
    
    dataset = [[{"role":"user","content":prompt}] for prompt in prompts]

    if verbose: 
        print(pipe.model.device)
        print(pipe.model.dtype)
        print("Model on device : ", pipe.model.device ,torch.cuda.get_device_name(pipe.model.device))#maybe dont work if on multiple devices
    start = time.process_time()

    res = [] 
    #res = [out for out in tqdm(pipe(dataset))]
    for out, file in zip(tqdm(pipe(dataset)), files): 
        save(out[0]['generated_text'], run, file, model_id)
                                   
    end = time.process_time()
    print("time : ", end-start)
    return res