from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset
from transformers import pipeline
import torch
import time

def inference(dataset,
              devices,
              model_id,
              hf_token,
              verbose=False,
              **kwargs):

    if any(["8000" in d for d in devices]):
        attn_implementation = 'sdpa'
    else : 
        attn_implementation = 'flash_attention_2'

    max_new_tokens=8192
    

    print("Launching : ", model_id)

    pipe = pipeline("text-generation",  
                    model=model_id,
                    model_kwargs={'attn_implementation':attn_implementation,
                                  'torch_dtype':torch.bfloat16,},
                    device_map="auto",
                    token=hf_token,
                    max_new_tokens=max_new_tokens,
                    return_full_text =False,
                    add_special_tokens=True)

    terminators = [
            pipe.tokenizer.eos_token_id,
            pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

    pipe.model.generation_config.pad_token_id = pipe.tokenizer.eos_token_id
    if verbose: 
        print(pipe.model.device)
        print(pipe.model.dtype)
        print("Model on device : ", pipe.model.device ,torch.cuda.get_device_name(pipe.model.device))#maybe dont work if on multiple devices

    start = time.process_time()
    res = [out[0]['generated_text'] for out in tqdm(pipe(dataset,
                                    eos_token_id=terminators,
                                    do_sample=True,
                                    temperature=0.6,
                                    top_p=0.9,))]
    end = time.process_time()
    print("time : ", end-start)
    return res