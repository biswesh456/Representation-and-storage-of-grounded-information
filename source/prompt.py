import pandas as pd
from glob import glob
import datetime
from transformers import pipeline, AutoTokenizer, AutoModel
from utils import get_files, save
import utils
from inference import Summary
from tqdm import tqdm
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
import torch
import json 
import chromadb

def get_dialog(df,dataset_name, user="A", img=True, return_messages=False):
    if "spot" in dataset_name:
        return get_dialog_spot(df, user, img, return_messages)
    else:
        return get_dialog_meetup(df, user, img, return_messages)

def get_dialog_spot(df, user="A", img=True,return_messages=False):
    prompt = ""
    offset =  datetime.timedelta(0)
    if return_messages:
        messages, times, users = [],[],[]
    for index, row in df.iterrows():
        if type(row["start_time"]) == float :
            offset += datetime.timedelta(0,4)
            row["start_time"] = pd.to_datetime(df.loc[index-1, "start_time"], format='%H:%M:%S')
        curr_time = str(pd.to_datetime(row["start_time"], format='%H:%M:%S')+offset).split(" ")[-1]
        
        if index == len(df)-3:
            answer = row["utterance"]
            print("Answer", answer)
            if return_messages:
                return messages, times, users, answer, row["speaker"], curr_time
            break

        prompt += "[" + curr_time + "]"  + "[" + row["speaker"]+ "]" +" "+ row["utterance"] + "\n"
        if return_messages:
            messages += [row["utterance"]]
            times += [curr_time]
            users += [row["speaker"]]

    return prompt, answer

def get_dialog_meetup(df, user="A", img=True,return_messages=False):
    user = df["user"].array[-1]# handles the right selection of the user
    image_df = pd.read_csv("../../LLM-Grounding-Study/data/image_descriptions_all_final.csv")

    image_descriptions_dict = {}
    for i in range(len(image_df)):
        image_descriptions_dict[image_df.iloc[i]['image_path']] = image_df.iloc[i]['description']

    if return_messages:
        messages = []
        users = []
        times = []
    prompt = ""
    second = False
    lastindex = 0
    lastimage = ""
    lasttime = pd.to_datetime(df.loc[0, "time"].split(" ")[-1], format='%H:%M:%S')
    time_offset = datetime.timedelta(0)
    answer = ""
    for index, row in df.iterrows():
        temp_image = row[user + "_inst"]

        #if NaN (new otturances, happens for the last rows)
        if type(row["time"]) == float:
            time = df.loc[lastindex, "time"].split(" ")[-1]
            time = pd.to_datetime(time, format='%H:%M:%S') + datetime.timedelta(0,4)
            #just to not have a second offset added (ugly code)
            if not second: 
                df.loc[index, "time"] = str(time + time_offset).split(" ")[-1]
                second = True
            else : 
                second = False
                df.loc[index, "time"] = str(time).split(" ")[-1]
            if index != df.index[-1]:
                prompt += "["+ df.loc[index, "time"] + "] "
                if return_messages:
                    times += [df.loc[index, "time"]]
            else : 
                answer += "["+ df.loc[index, "time"] + "] "
                if return_messages:
                        answer_time = time
                
        else : 
            time = pd.to_datetime(df.loc[index,"time"].split(" ")[-1], format='%H:%M:%S')
            #when there is a change of dialog
            #case when date not synced
            if time <= lasttime :
                time_offset += datetime.timedelta(0,4)
                time += time_offset
                time = str(time).split(" ")[-1]
                if index != df.index[-1]:
                    prompt += "["+ time + "] "
                    if return_messages:
                        times += [time]
            #normal utterance where we add the time offset occuring
            else : 
                time += time_offset
                time = str(time).split(" ")[-1]
                if index != df.index[-1]:
                    prompt += "["+ time + "] "
                    if return_messages:
                        times += [time]
        
        if index != df.index[-1]:
            prompt += row["user"] + ": "
            prompt += row["msg"]
            if return_messages:
                users += [row["user"]]
                messages += [row["msg"]]
        else : 
            
            answer += row["user"] + ": "
            answer += row["msg"]
            if return_messages:
                answer_time = times.pop(-1)
                answer_user = row["user"]
                answer = row["msg"]
        
        prompt += "\n"
        if img :
            if lastimage != temp_image and type(temp_image) == str:
                    prompt += "<Image "+ user +"> " + image_descriptions_dict[temp_image.split("/")[-1]] + " <Image "+ user +"> "
                    lastimage = temp_image
                    prompt += "\n"
                    if return_messages:
                        messages += ["<Image "+ user +"> " + image_descriptions_dict[temp_image.split("/")[-1]] + " <Image "+ user +"> "]
                        users += [user]
                        times += ["-1"]

        lasttime = pd.to_datetime(df.loc[index,"time"].split(" ")[-1], format='%H:%M:%S')
        lastindex = index
        
    if return_messages:
        return messages, times, users, answer, answer_user, answer_time
    return prompt, answer


def get_start_prompt(processing, dataset_name):

    if "spot" in dataset_name:
        prompt = "Instructions : Here is a conversation between two participants A and B who have been provided with two images that are slightly different. Each participant can only look at their own image. In order to finish the task they need to discuss with each other and come up with the differences in both of their images. Once they are confident of finding all the differences in their images they move to the next image. The goal of the game is to find all the differences in the pair of images that the participants are provided. Every utterance from A or B is preceded with a timestamp closed under brackets. The utterances also sometimes include information inside angular brackets <> which means that those words were spoken in lower volume.\nFollowing is the dialog history :\n"

        if processing == "summary":
            prompt = "Instructions : Here is a summary between two participants A and B who have been provided with two images that are slightly different. Each participant can only look at their own image. In order to finish the task they need to discuss with each other and come up with the differences in both of their images. Once they are confident of finding all the differences in their images they move to the next image. The goal of the game is to find all the differences in the pair of images that the participants are provided. Every utterance from A or B is preceded with a timestamp closed under brackets.\nFollowing is the summary with the last utterances :\n"
        if processing == "rag" or processing == "local-answer":
            prompt = "Instructions : Here are the retrieved utturances of a conversations between two participants A and B who have been provided with two images that are slightly different. Each participant can only look at their own image. In order to finish the task they need to discuss with each other and come up with the differences in both of their images. Once they are confident of finding all the differences in their images they move to the next image. The goal of the game is to find all the differences in the pair of images that the participants are provided. Every utterance from A or B is preceded with a timestamp closed under brackets. The utterances also sometimes include information inside angular brackets <> which means that those words were spoken in lower volume.\nFollowing is the retrieved utterances :\n"
        return prompt


    prompt = "Instructions : Here is a conversation between two Participants A and B who are in a virtual space that has lots of different rooms that are depicted with images. Each room has a type (such as kitchen, bathroom, bedroom, etc.).  The participants are initially located in different rooms. The goal of the game is for the two participants to locate themselves in the same room. In order to achieve this goal, the participants communicate with one another by text and describe the room they find themselves in. On the basis of those descriptions, they move to different rooms and describe their new room to the other participant. The game ends when the two participants find themselves in the same room. We translated the images that the participants saw into text. That description of the room is provided below as soon as a participant enters a given room. The current room description of User A starts with a token <Image A> and the current room description of User B starts with a token <Image B>. Every utterance from A or B is preceded with a timestamp closed under brackets.\nFollowing is the dialog history along with image descriptions :\n"
    if processing == "summary":
        prompt = "Instructions : Here is a summary of a conversation between two Participants A and B who are in a virtual space that has lots of different rooms that are depicted with images. Each room has a type (such as kitchen, bathroom, bedroom, etc.).  The participants are initially located in different rooms. The goal of the game is for the two participants to locate themselves in the same room. In order to achieve this goal, the participants communicate with one another by text and describe the room they find themselves in. On the basis of those descriptions, they move to different rooms and describe their new room to the other participant. The game ends when the two participants find themselves in the same room. The current room description of User A starts with a token <Image A> and the current room description of User B starts with a token <Image B>. Every utterance from A or B is preceded with a timestamp closed under brackets.\nFollowing is the summary with the last utterances :\n"
    if processing == "rag" or processing == "local-answer":
        prompt = "Instructions : Here are the retrieved utturances of a conversation between two Participants A and B who are in a virtual space that has lots of different rooms that are depicted with images. Each room has a type (such as kitchen, bathroom, bedroom, etc.). The participants are initially located in different rooms. The goal of the game is for the two participants to locate themselves in the same room. In order to achieve this goal, the participants communicate with one another by text and describe the room they find themselves in. On the basis of those descriptions, they move to different rooms and describe their new room to the other participant. The game ends when the two participants find themselves in the same room. Every utterance from A or B is preceded with a timestamp closed under brackets.\nFollowing is the retrieved utturances :\n"

    return prompt



def get_end_prompt(user="A"):
    # here probably just as him to just answer the question 
    prompt = "\nPlease answer the question of the user by providing only the next utterance and by keeping in mind that it's a spoken conversation."
    return prompt


def make_prompt(df, tokenizer, model_id, file, processing, CoT, dataset_name=None):
    prompt = get_start_prompt(processing=processing, dataset_name=dataset_name)
    print(file)
    dialog, answer = get_dialog(df, dataset_name)
    
    if CoT : 
        end_prompt = "\nThink between <thinking> tags before you write the answer. First think through what is the answer to the question of the user. Then using your analysis answer the question of the user by formatting the answer as the next utterance and by keeping in mind it's a dialog. Very important don't forget to format your thoughts between <thinking> tags."

    else:
        end_prompt = get_end_prompt()


    if processing is None:
        prompt += dialog
    else: 
        if processing == "windowed":
            tokenized = tokenizer([prompt, dialog, end_prompt], return_offsets_mapping=True)
            offsets = tokenized["offset_mapping"]
            input_ids = tokenized["input_ids"]
            prompt_len = len(input_ids[0]) + len(input_ids[1]) + len(input_ids[2])
            split_tok = tokenizer("\n")['input_ids'][1]
            if prompt_len > 1840:
                offset_size = prompt_len - 1850
                indexes = [i for i,t in enumerate(input_ids[1]) if (t == split_tok) and i > offset_size]
                #indexes = [i for i in indexes if (i > offset_size)]
                #edge case where prompt_len too close to 1850   
                if indexes == []:
                    for i,t in enumerate(input_ids[1]):
                        if t == split_tok:
                            indexes = [i]
                            break
                dialog = dialog[offsets[1][indexes[0]][-1]:]
            prompt += dialog
            #generate summaries if not existing
        if processing == "summary":
            with open("../data/Summary/"+dataset_name+ "/" + utils.MODELS[model_id] + "/" + file.split("/")[-1].split(".")[0] + ".json") as f:
                jdict = json.load(f)
                summary = jdict["completion"] 
                f.close()
            prompt = get_start_prompt(processing=processing, dataset_name=dataset_name) + summary
        if processing == "rag":
            with open("../data/RAG/"+dataset_name+ "/"+ file.split("/")[-1].split(".")[0] + ".txt") as f:
                rag = f.read()
                f.close()
            prompt = get_start_prompt(processing=processing, dataset_name=dataset_name) + rag
        if processing == "rag_bm25":
            #TODO retrieve bm25
            with open("../data/RAGBM25/"+dataset_name+ "/"+ file.split("/")[-1].split(".")[0] + ".txt") as f:
                rag = f.read()
                f.close()
            prompt = get_start_prompt(processing=processing, dataset_name=dataset_name) + rag

        if processing == "only_dialog":
            return dialog,answer
        
        if processing == "memgpt":
            split_dialog = dialog.split("\n")
            if "\n" in split_dialog : 
                split_dialog.remove("\n")
            if "" in split_dialog : 
                split_dialog.remove("")
            query = split_dialog[-2]
            split_dialog = split_dialog[:-2]
            dialog = ""
            for split in split_dialog:
                dialog += split + "\n"
            dialog = dialog[:-1]
            return prompt, dialog, query
        if processing == "memgpt-messages":
            return get_dialog(df,dataset_name, return_messages=True)
        
        if processing == "local-answer":
            prompt = get_start_prompt(processing=processing, dataset_name=dataset_name)
            messages, times, users, answer, answer_user, answer_time = get_dialog(df, dataset_name, return_messages=True)
            if "meetup" in dataset_name:
                splits = file.split("/")
                local_df = pd.read_csv("../data/meetup_target_lines/"+splits[-2]+".csv")
                nbline = int(local_df.loc[local_df["file"] == splits[-1]]["line"].iloc[0])
            else: 
                nbline = int(df.iloc[-1]["utterance"])
            offset = 0
            for i,m in enumerate(messages):
                if "<image" in m :
                    offset += 1
                    continue
                if (i - 4 - offset > nbline - 4) and (i+4 - offset < nbline) + 4:
                    prompt += m + "\n"
    prompt += end_prompt
    
    return prompt, answer


def load_prompt(files, tokenizer, model_id, processing=None, CoT=False, dataset_name=None):
    prompts = []
    answers = []

    if processing == "memgpt":
        preprompts = []
        for f in files : 
            #print(f)
            df = pd.read_csv(f)
            preprompt, prompt, answer = make_prompt(df, tokenizer, model_id, f, processing, CoT, dataset_name)
            #print(len(tokenizer(prompt)['input_ids']))
            preprompts += [preprompt]
            prompts += [prompt]
            answers += [answer]
        return preprompts, prompts, answers
    
    if processing == "memgpt-messages":
        l_messages, l_times, l_users, l_answer, l_answer_user, l_answer_time  = ([],[],[],[],[],[])
        for f in files: 
            df = pd.read_csv(f)
            messages, times, users, answer, answer_user, answer_time = make_prompt(df, tokenizer, model_id, f, processing, CoT, dataset_name)
            l_messages += [messages]
            l_times += [times]
            l_users += [users]
            l_answer += [answer]
            l_answer_user += [answer_user]
            l_answer_time += [answer_time]
        return l_messages, l_times, l_users, l_answer, l_answer_user, l_answer_time 
    for f in files : 
        #print(f)
        df = pd.read_csv(f)
        prompt, answer = make_prompt(df, tokenizer, model_id, f, processing, CoT, dataset_name)
        #print(len(tokenizer(prompt)['input_ids']))
        prompts += [prompt]
        answers += [answer]
    return prompts, answers

def make_summaries(pipe, files, **parameters):
    n = 5
    if "run" in parameters:
        run = parameters.pop("run")
    if "model_id" in parameters:
        model_id = parameters.pop("model_id")
    if "dataset_name" in parameters['kwargs']:
        dataset_name = parameters['kwargs'].pop("dataset_name")
    start = get_start_prompt(processing="noprocessing", dataset_name=dataset_name)
    end = "\nSummarize the conversation without missing any information in less than 200 words."
    files = get_files(run, model_id, optional_arg=run.__name__.split('.')[-1], dataset_name=parameters["dataset_name"])
    params = dict(parameters)
    dataset_name = params.pop("dataset_name")
    for f in tqdm(files) : 
        df = pd.read_csv(f)
        dialog, answer = get_dialog(df, dataset_name)
        split_dialog = dialog.split("\n")[-n-3:]
        last_n = ""
        for d in split_dialog:
            last_n += d + "\n"
        last_n = last_n[:-3]
        summary = pipe([{"role":"user","content":start+dialog+end}], max_new_tokens=300, **params)
        save(summary[0]['generated_text'] +"\n"+ last_n, run, f, model_id, run.__name__.split(".")[-1], dataset_name=dataset_name)


def make_rag(pipe, files, **parameters):
    n = 5
    if "run" in parameters:
        run = parameters.pop("run")
    if "model_id" in parameters:
        model_id = parameters.pop("model_id")
    if "overlap" in parameters['kwargs']:
        overlap = parameters['kwargs'].pop("overlap")
    if "chunk_size" in parameters['kwargs']:
        chunk_size = parameters['kwargs'].pop("chunk_size")
    if "dataset_name" in parameters['kwargs']:
        dataset_name = parameters['kwargs'].pop("dataset_name")
    if "attn_implementation" in parameters:
        attn_implementation = parameters.pop("attn_implementation")
    start = get_start_prompt(processing="rag", dataset_name=dataset_name)
    end = get_end_prompt()
    files = get_files(run, "", optional_arg=run.__name__.split('.')[-1], dataset_name=parameters["dataset_name"])
    print("Processing RAG, files : ", len(files))
    for f in tqdm(files) : 
        df = pd.read_csv(f)
        dialog, answer = get_dialog(df, dataset_name, img=False)
        split_dialog = dialog.split("\n")
        if "\n" in split_dialog : 
            split_dialog.remove("\n")
        if "" in split_dialog : 
            split_dialog.remove("")
        query = split_dialog[-2]
        split_dialog = split_dialog[:-1]
        chunks_overlapped = [split_dialog[i: i + chunk_size]for i in range(0,len(split_dialog) - chunk_size + 1, overlap)]
        documents = []
        chunks = []
        for chunk in chunks_overlapped:
            temp_chunk = ""
            for otturence in chunk:
                temp_chunk += otturence + "\n"
            temp_chunk = temp_chunk[:-2]

            chunks += [temp_chunk]
            documents += [Document(page_content=temp_chunk)]


        if run.__name__.split(".")[-1] == "RAGBM25":
            retriever = BM25Retriever.from_documents(documents, k=3)
            docs = [d.page_content for d in retriever.invoke(query)]
        else : 
            chroma_client = chromadb.PersistentClient(path="../data/RAGdatabase/db"+"/"+parameters["dataset_name"])
            if "spot" in dataset_name : 
                file_name = "_".join(f.split("/")[-1].split(".")[0].split("_")[1:])
            else: 
                file_name = f.split("/")[-1].split(".")[0]
            colname = file_name
            if colname in [c.name for c in chroma_client.list_collections()]:
                continue

            collection = chroma_client.create_collection(name=file_name)
            parameters["autoset_attn_implementation"] =True

            collection.add(
                documents=chunks,
                embeddings=generate_embeddings(chunks, "document", **parameters),
                ids=["id" + str(i) for i in range(len(chunks))]
            )
            
            db = Chroma(client=chroma_client, collection_name=colname) 

            docs = db.similarity_search_by_vector_with_relevance_scores(generate_embeddings(query, "query", **parameters), k=3)
            docs = [d[0].page_content for d in docs]
        prompt = ""
        for d in docs : 
            prompt += d + "\n"

        prompt += query
        save(prompt, run, f, "", run.__name__.split(".")[-1], dataset_name=parameters["dataset_name"])

    #if attn_implementation :
        #parameters["attn_implementation"] = attn_implementation

def pre_generate(pipe, files, **parameters):

    run = parameters["run"]
    str_run = run.__name__.split(".")[-1]

    if str_run == "Summary":
        make_summaries(pipe, files, **parameters)
    if str_run == "RAG":
        make_rag(pipe, files, **parameters)
    if str_run == "RAGBM25":
        parameters
        make_rag(pipe, files, **parameters)

def generate_embeddings(documents, pre=None, **parameters):
    prefixes = {"query":"Given the question from the user, retrieve relevant otturances that answer the question: ",
               "document" :"otturances: "}
    prefix = prefixes[pre]
    documents = [f"{prefix} {document}" for document in documents]
    #tokenizer = AutoTokenizer.from_pretrained('nvidia/NV-Retriever-v1')
    #model = AutoModel.from_pretrained('nvidia/NV-Retriever-v1', trust_remote_code=True, torch_dtype=torch.float16)
    tokenizer = parameters["kwargs"]["rag_tokenizer"]
    model = parameters["kwargs"]["rag_model"]
    tok_documents = tokenizer(documents, padding=True, truncation=True, return_tensors='pt').to("cuda")
    if "NV" in str(parameters["kwargs"]["rag_model"]):
        model.model._attn_implementation = ""
    model.to("cuda")
    with torch.no_grad():
        embeddings_queries = model(**tok_documents)
    return embeddings_queries.tolist()


#def retrieve_bm25(documents)