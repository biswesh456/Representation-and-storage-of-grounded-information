from .models import llama, gemma

def inference(files, **parameters):
    
    model_id = parameters['model_id']

    if 'llama' in model_id:
        res = llama.inference(files=files, verbose=True, **parameters)
    if "gemma" in model_id:
        res = gemma.inference(files=files, verbose=True, **parameters)

    return res