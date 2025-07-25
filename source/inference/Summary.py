from .models import llama, gemma, qwenQwQ


def inference( files, **parameters):
    
    model_id = parameters['model_id']
    #modify max_length to have toggle sliding window or not
    parameters['max_length']=4096
    parameters['max_length']=2048
    parameters['processing']='summary'
    
    if 'llama' in model_id:
        res = llama.inference(files=files, verbose=True, **parameters)
    if "gemma" in model_id:
        res = gemma.inference(files=files, verbose=True, **parameters)
    if "QwQ" in model_id:
        res = qwenQwQ.inference(files=files, verbose=True, **parameters)

    return res