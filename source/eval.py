
if __name__ == '__main__':

    import os
    os.environ['HF_HOME'] = '/data/almanach/user/cdearauj/models/'
    from evaluate import load
    from glob import glob
    import numpy as np
    import pandas as pd
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams
    import torch
    torch.multiprocessing.set_start_method('spawn')
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import json 
    import gc 
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
                    #print(gens)
                    with open(gens, "r") as file_model:
                        generation = file_model.read()
                        file_model.close()
                    models[temp_model][temp_d][relation][gens.split("/")[-1].split(".")[0]] = generation.strip()
                    #print(models[temp_model][relation][gens.split("/")[-1].split(".")[0]])
    labels = {}
    questions = {}
    exact_labels = {}
    for d in directories:
        temp_d = d.split("/")[-1]
        labels[temp_d] = {}
        exact_labels[temp_d] = {}
        questions[temp_d] = {}
        #print(models)
        files = glob(d+"/*.txt")
        for file in files : 
            temp_file = file.split("/")[-1].split(".")[0]
            with open(file, "r") as f:
                lines = f.readlines()
                label = lines[0][:-1]
                exact_label = lines[1]
                f.close()
            df = pd.read_csv("../data/meetup_target/"+ file.split("/")[3] + "/" +file.split("/")[-1].split(".")[0]+".csv", sep=",")
            questions[temp_d][temp_file] = df.iloc[-2]["msg"]
            labels[temp_d][temp_file] = label
            exact_labels[temp_d][temp_file] = exact_label
    llm = LLM("Qwen/Qwen2.5-32B-Instruct", tensor_parallel_size=2)
    guided_decoding_params = GuidedDecodingParams(choice=["True", "False"])
    sampling_params = SamplingParams(guided_decoding=guided_decoding_params)

    with open("../data/results.csv", "w") as f:
        f.write("model,relation,inference,precision,recall,f1,EM,fEM,Judge")
        f.close()
    for model in models:
        print(model)
        print("models number : ", len(models[model]))
        for relation in models[model]:
            print("relation",relation)
            #print("relation number : ",len(models[model][relation]))
            for infer in models[model][relation]:
                if os.path.exists("../runs/results_meetup_target_json/"+relation+ "/" + infer + "/"+ model + "/" + models[model][relation][infer][list(models[model][relation][infer].keys())[0]] + '.json'):
                        continue
                print(infer)
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
                prompts = []
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
                    if "QwQ" in model:
                        pred = models[model][relation][infer][file].split("</think>")[-1]
                        #trying not to run out of memory because of some model outputs
                        pred = pred[-2000:]
                        predictions += [pred]
                    else :
                        predictions += [models[model][relation][infer][file]]
                    predictions[-1] = predictions[-1].split(":")[-1]
                    references += [labels[relation][file].split(":")[-1]]
                    
                    exact_references += [exact_labels[relation][file]]
                    fEM += [fuzzy_EM(predictions[-1], exact_references[-1])]
                    prompts += ["You are a judge who has to decide whether the generated answer for a question is correct or not. You will be provided with the correct answer and the question. Based on these two informations you need to decide whether the generated answer has the same meaning as the correct answer. If they have the same meaning then answer True otherwise answer False. Here is the question : "+questions[relation][file]+ ". Here is the correct answer : " + labels[relation][file].split(":")[-1] + ". Here is the generated answer : " + predictions[-1] + ". Please provide True if the generated answer has the same meaning as the correct answer otherwise provide False."]
                outputs = llm.generate(
                    prompts=prompts,
                    sampling_params=sampling_params,
                    use_tqdm=True
                )
                judgement = len([ 1 for o in outputs if o.outputs[0].text == "True"])/ len(prompts)
                #TODO include in json per file metrics

                for i,file in enumerate(models[model][relation][infer]):
                    with open("../runs/meetup_target_json/"+relation+ "/" + infer + "/"+ model + "/" + file + '.json') as json_file:
                        data = json.load(json_file)
                        data["judge"] = outputs[i].outputs[0].text
                        data["label"] = labels[relation][file].split(":")[-1]
                        data["generated"] = predictions[i]
                    if not os.path.exists("../runs/results_meetup_target_json/"+relation+ "/" + infer + "/"+ model + "/"):
                        os.makedirs("../runs/results_meetup_target_json/"+relation+ "/" + infer + "/"+ model + "/")
                    with open("../runs/results_meetup_target_json/"+relation+ "/" + infer + "/"+ model + "/" + file + '.json', "w") as outfile:
                        json.dump(data, outfile, indent=4)
                #"%.2f" % judgement
    #llm.to("cpu")
    #del llm
    #torch.cuda.empty_cache()
    #gc.collect()

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
                prompts = []
                for file in models[model][relation][infer]:
                    if "CoT" in infer :
                        pred = models[model][relation][infer][file].split("</thinking>")
                        if len(pred) > 1:
                            pred = pred[-1]
                        else :
                            pred = models[model][relation][infer][file].split("<thinking>")[-1]
                        #trying not to run out of memory because of some model outputs
                        pred = pred[-2000:]
                        predictions += [pred]
                    if "QwQ" in model:
                        pred = models[model][relation][infer][file].split("</think>")[-1]
                        #trying not to run out of memory because of some model outputs
                        pred = pred[-2000:]
                        predictions += [pred]
                    else :
                        predictions += [models[model][relation][infer][file]]
                    references += [labels[relation][file].split(":")[-1]]
                    exact_references += [exact_labels[relation][file]]
                    fEM += [fuzzy_EM(predictions[-1], exact_references[-1])]
                    prompts += ["You are a judge who has to decide whether the answer provided to a question is correct or not. You will be provided with the correct answer and the question. Based on these two informations you need to decide whether the provided answer has the same meaning as the correct answer. If they have the same meaning then answer True otherwise answer False. Here is the question : "+questions[relation][file]+ ". Here is the correct answer : " + labels[relation][file].split(":")[-1] + ". Here is the provided answer : " + predictions[-1]]
                    #prompts += ["Here is a question with the label answer and the user answer, tell me if the user's answer is correct following the label answer: question: " + questions[relation][file]+ "\nlabel answer: " + labels[relation][file] + "\nuser answer: "+ predictions[-1]]
            
                #TODO include in json per file metrics
                results = bertscore.compute(predictions=predictions, references=references, model_type="microsoft/deberta-xlarge-mnli",num_layers=40,lang="en", batch_size=32)
                em_results = exact_match.compute(references=exact_references,predictions=predictions,ignore_case=True,ignore_punctuation=True)
                
                judge_mean = []
                for i,file in enumerate(models[model][relation][infer]):
                    with open("../runs/results_meetup_target_json/"+relation+ "/" + infer + "/"+ model + "/" + file + '.json') as json_file:
                        data = json.load(json_file)
                        data["bertscore_precision"] = results['precision'][i]
                        data["bertscore_recall"] = results['precision'][i]
                        data["bertscore_f1"] = results['precision'][i]
                        
                        judge_mean += [1 if data["judge"] == "True" else 0]
                    if not os.path.exists("../runs/results_meetup_target_json/"+relation+ "/" + infer + "/"+ model + "/"):
                        os.makedirs("../runs/results_meetup_target_json/"+relation+ "/" + infer + "/"+ model + "/")
                    with open("../runs/results_meetup_target_json/"+relation+ "/" + infer + "/"+ model + "/" + file + '.json', "w") as outfile:
                        json.dump(data, outfile, indent=4)


                print("Judge mean : " +model + "," + relation + ","+infer  + "%.2f" % np.mean(judge_mean))
                with open("../data/results.csv", "a") as f:
                    f.write("\n"+model + ","+relation+","+infer+","+ str("%.2f" % np.mean(results['precision'])) + "," + "%.2f" % np.mean(results['recall'])+ "," + "%.2f" % np.mean(results['f1'])+ "," + "%.2f" % round(em_results["exact_match"])+ "," + "%.2f" % np.mean(fEM) + "," + "%.2f" % np.mean(judge_mean))
                    f.close()
                #print(model, relation, infer, results)
                print(infer)
                
                print("p:", "%.2f" % np.mean(results['precision']), "r:", "%.2f" %np.mean(results['recall']), "f1:", "%.2f" %np.mean(results['f1']))
        torch.cuda.empty_cache()
