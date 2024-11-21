import os
os.environ['HF_HOME'] = '/data/almanach/user/cdearauj/models/'
from evaluate import load
from glob import glob
import numpy as np


import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def fuzzy_EM(pred, ref):
    pred = pred.lower()
    ref = ref.lower()
    ref = ref.split(" ")
    nb_union = [True if r in pred else False for r in ref]
    return np.mean(nb_union)

bertscore = load("bertscore")
exact_match = load("exact_match")

models = {}
predictions = glob("../data/meetup_target/")
directories = [ f.path for f in os.scandir("../data/answers/") if f.is_dir() ]
run_dir = "../runs/meetup_target/"
for d in directories:
   
    temp_d = d.split("/")[-1]
    sub_dirs = [f.path for f in os.scandir(run_dir + temp_d + "/") if f.is_dir() ]
    #print(sub_dirs)
    for sub in sub_dirs:
        model_dirs = [f.path for f in os.scandir(sub + "/") if f.is_dir() ]
        #print(model_dirs)
        for model in model_dirs:
            temp_model = model.split("/")[-1]
            if temp_model not in models:
                models[temp_model]={}
            if temp_d not in models[temp_model]:
                models[temp_model][temp_d] = {}
            relation = sub.split("/")[-1]
            models[temp_model][temp_d][relation] = {}
            for gens in glob(model+"/*.txt"):
                with open(gens, "r") as file_model:
                    generation = file_model.read()
                    file_model.close()
                models[temp_model][temp_d][relation][gens.split("/")[-1].split(".")[0]] = generation[:-1]
                #print(models[temp_model][relation][gens.split("/")[-1].split(".")[0]])
labels = {}
exact_labels = {}
for d in directories:
    temp_d = d.split("/")[-1]
    labels[temp_d] = {}
    exact_labels[temp_d] = {}
    #print(models)
    files = glob(d+"/*.txt")
    for file in files : 
        temp_file = file.split("/")[-1].split(".")[0]
        with open(file, "r") as f:
            lines = f.readlines()
            label = lines[0][:-1]
            exact_label = lines[1]
            f.close()
        labels[temp_d][temp_file] = label
        exact_labels[temp_d][temp_file] = exact_label


with open("../data/results.csv", "w") as f:
    f.write("model,relation,inference,precision,recall,f1,EM,fEM")
    f.close()
for model in models:
    print(model)
    print("models number : ", len(models[model]))
    for relation in models[model]:
        print("relation",relation)
        #print("relation number : ",len(models[model][relation]))
        for infer in models[model][relation]:
            #labels[relation][]
            nb_files_model = len(models[model][relation][infer])
            nb_files_labels = len(labels[relation])
            set_models = set(models[model][relation][infer])
            set_labels = set(labels[relation])
            if len(set_labels - set_models) != 0:
                #We don't compute the metric because the inference isn't complete !
                continue
            predictions = []
            references = []
            exact_references = []
            fEM = []
            for file in models[model][relation][infer]:
                if "CoT" in infer :
                    pred = models[model][relation][infer][file].split("</thinking>")
                    if len(pred) > 1:
                        pred = pred[-1]
                    else :
                        pred = models[model][relation][infer][file].split("<thinking>")[-1]
                    #trying not to run out of memory because of some model outputs
                    pred = pred[-6000:]
                    predictions += [pred]
                else :
                    predictions += [models[model][relation][infer][file]]
                references += [labels[relation][file]]
                exact_references += [exact_labels[relation][file]]
                fEM += [fuzzy_EM(predictions[-1], exact_references[-1])]
                
            results = bertscore.compute(predictions=predictions, references=references, model_type="microsoft/deberta-xlarge-mnli",num_layers=40,lang="en", batch_size=32)
            em_results = exact_match.compute(references=exact_references,predictions=predictions,ignore_case=True,ignore_punctuation=True)
            
            with open("../data/results.csv", "a") as f:
                f.write("\n"+model + ","+relation+","+infer+","+ str("%.2f" % np.mean(results['precision'])) + "," + "%.2f" % np.mean(results['recall'])+ "," + "%.2f" % np.mean(results['f1'])+ "," + "%.2f" % round(em_results["exact_match"])+ "," + "%.2f" % np.mean(fEM))
                f.close()
            #print(model, relation, infer, results)
            print(infer)
            
            print("p:", "%.2f" % np.mean(results['precision']), "r:", "%.2f" %np.mean(results['recall']), "f1:", "%.2f" %np.mean(results['f1']))
    torch.cuda.empty_cache()



#Bertscore tests       
"""   
predictions = ["hello there", "general kenobi"]
references = ["hello there", "general kenobi"]
results = bertscore.compute(predictions=["I think it was a landscape with trees"], references=["It was a long painting"], model_type="microsoft/deberta-xlarge-mnli",num_layers=40,lang="en")

print(results)





tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-v2-m3')
model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-v2-m3')
model.eval()

pairs = [["What color is the moon?", "It is white"],["What color is the moon?", "The moon is white"],["I think it was a landscape with trees", '"It was a long painting"'],['A red pineapple', "A reddish pineapple"],['what is panda?', 'A red pineapple']]
with torch.no_grad():
    inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
    scores = torch.sigmoid(model(**inputs, return_dict=True).logits.view(-1, ).float())

    print(scores)
""" 