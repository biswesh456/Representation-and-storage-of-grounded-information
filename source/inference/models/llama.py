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
              **kwargs):

    

    max_new_tokens=8192
    

    print("Launching : ", model_id)


    pipe = pipeline("text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    token=hf_token,
                    max_length=max_length,
                    device_map="auto",
                    #max_new_tokens=max_new_tokens,
                    return_full_text =False,
                    add_special_tokens=True)
    terminators = [
            pipe.tokenizer.eos_token_id,
            pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

    pipe.model.generation_config.pad_token_id = pipe.tokenizer.eos_token_id


    prompts, answers = prompting.load_prompt(files, tokenizer=pipe.tokenizer, processing=processing) #TODO connect to real data
    
    dataset = [[{"role":"user","content":prompt}] for prompt in prompts]

    if verbose: 
        print(pipe.model.device)
        print(pipe.model.dtype)
        print("Model on device : ", pipe.model.device ,torch.cuda.get_device_name(pipe.model.device))#maybe dont work if on multiple devices

    start = time.process_time()
    for out,file in zip(tqdm(pipe(dataset,eos_token_id=terminators,do_sample=True,temperature=0.6,top_p=0.9,)), files):
        save(out[0]['generated_text'], run,file, model_id)
    
    #res = [out[0]['generated_text'] for out in tqdm(pipe(dataset,
    #                                eos_token_id=terminators,
    #                                do_sample=True,
    #                                temperature=0.6,
    #                                top_p=0.9,))]
    end = time.process_time()
    print("time : ", end-start)
    return []