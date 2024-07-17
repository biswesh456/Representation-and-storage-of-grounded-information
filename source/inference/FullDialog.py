from .models import llama, gemma

def inference(dataset, **parameters):
    
    model_id = parameters['model_id']

    if 'llama' in model_id:
        res = llama.inference(dataset=dataset, verbose=True, **parameters)
    if "gemma" in model_id:
        res = gemma.inference(dataset=dataset, verbose=True, **parameters)

    return res