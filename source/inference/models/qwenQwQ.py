from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset
from transformers import pipeline
import torch
import time
import prompt as prompting
from utils import get_files,save
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
              CoT=False,
              **kwargs):

    

    max_new_tokens=8192
    

    print("Launching : ", model_id)

    if "spot" in kwargs["dataset_name"]:
        pipe = pipeline("text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        token=hf_token,
                        max_new_tokens=300,
                        device_map="auto",
                        return_full_text =False,
                        add_special_tokens=True,
                        )
    else : 
        pipe = pipeline("text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        token=hf_token,
                        max_length=max_length,
                        device_map="auto",
                        return_full_text =False,
                        add_special_tokens=True,
                        )

    parameters ={"do_sample":True,
                 "temperature":0.6,
                 "top_p":0.95,
                 "min_p":0,
                 "model_id":model_id,
                 "run":run,
                 "kwargs" : kwargs,
                 "dataset_name":kwargs["dataset_name"]}
    
    

    pipe.model.generation_config.pad_token_id = pipe.tokenizer.eos_token_id

    #generate the prompts 
    prompting.pre_generate(pipe, files, **parameters)

    prompts, answers = prompting.load_prompt(files, tokenizer=pipe.tokenizer, model_id=model_id, processing=processing, CoT=CoT, dataset_name=kwargs["dataset_name"]) #TODO connect to real data
    
    #print(prompts)
    #for a,f in zip(answers,files):
    #    save(a, run, f, model_id, optional_arg="answers", CoT=CoT, dataset_name=kwargs["dataset_name"])
        #print(a,f)

    #print(answers)
    dataset = [[{"role":"user","content":prompt}] for prompt in prompts]

    if verbose: 
        print(pipe.model.device)
        print(pipe.model.dtype)
        print("Model on device : ", pipe.model.device ,torch.cuda.get_device_name(pipe.model.device))#maybe dont work if on multiple devices

    start = time.process_time()
    for out,file in zip(tqdm(pipe(dataset, do_sample=True,temperature=0.6,top_p=0.95, min_p=0, repetition_penalty=1.1)), files):
        save(out[0]['generated_text'], run, file, model_id, CoT=CoT, dataset_name=kwargs["dataset_name"])
    
    end = time.process_time()
    print("time : ", end-start)
    return end-start