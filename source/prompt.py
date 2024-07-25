import pandas as pd
from glob import glob
import datetime
from transformers import pipeline
from utils import get_files, save
import utils
from inference import Summary
from tqdm import tqdm

def get_dialog(df, user="A"):
    image_df = pd.read_csv("../../LLM-Grounding-Study/data/image_descriptions_all_final.csv")

    image_descriptions_dict = {}
    for i in range(len(image_df)):
        image_descriptions_dict[image_df.iloc[i]['image_path']] = image_df.iloc[i]['description']

    prompt = ""
    second = False
    lastindex = 0
    lastimage = ""
    lasttime = pd.to_datetime(df.loc[0, "time"].split(" ")[-1], format='%H:%M:%S')
    time_offset = datetime.timedelta(0)
    answer = ""
    for index, row in df.iterrows():
        temp_image = row[user + "_inst"]

        #if NaN (new otturances)
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
            else : 
                answer += "["+ df.loc[index, "time"] + "] "
        else : 
            time = pd.to_datetime(df.loc[index,"time"].split(" ")[-1], format='%H:%M:%S')
            #when there is a change of dialog
            if time < lasttime :
                time_offset += lasttime - time
            else : 
                time += time_offset
                time = str(time).split(" ")[-1]
                if index != df.index[-1]:
                    prompt += "["+ time + "] "
        
        if index != df.index[-1]:
            prompt += row["user"] + ": "
            prompt += row["msg"]
        else : 
            answer += row["user"] + ": "
            answer += row["msg"]
        
        prompt += "\n"

        if lastimage != temp_image and type(temp_image) == str:
                prompt += "<Image "+ user +"> " + image_descriptions_dict[temp_image.split("/")[-1]] + " <Image "+ user +"> "
                lastimage = temp_image
                prompt += "\n"
        lasttime = pd.to_datetime(df.loc[index,"time"].split(" ")[-1], format='%H:%M:%S')
        lastindex = index
        
    return prompt, answer


def get_start_prompt(processing):
    if processing == "summary":
            prompt = "Instructions : Here is a summary of a conversation between two Participants A and B who are in a virtual space that has lots of different rooms that are depicted with images. Each room has a type (such as kitchen, bathroom, bedroom, etc.).  The participants are initially located in different rooms. The goal of the game is for the two participants to locate themselves in the same room. In order to achieve this goal, the participants communicate with one another by text and describe the room they find themselves in. On the basis of those descriptions, they move to different rooms and describe their new room to the other participant. The game ends when the two participants find themselves in the same room. We translated the images that the participants saw into text. That description of the room is provided below as soon as a participant enters a given room. The current room description of User A starts with a token <Image A> and the current room description of User B starts with a token <Image B>. Every utterance from A or B is preceded with a timestamp closed under brackets.\nFollowing is the dialog history along with image descriptions :\n"
    prompt = "Instructions : Here is a conversation between two Participants A and B who are in a virtual space that has lots of different rooms that are depicted with images. Each room has a type (such as kitchen, bathroom, bedroom, etc.).  The participants are initially located in different rooms. The goal of the game is for the two participants to locate themselves in the same room. In order to achieve this goal, the participants communicate with one another by text and describe the room they find themselves in. On the basis of those descriptions, they move to different rooms and describe their new room to the other participant. The game ends when the two participants find themselves in the same room. We translated the images that the participants saw into text. That description of the room is provided below as soon as a participant enters a given room. The current room description of User A starts with a token <Image A> and the current room description of User B starts with a token <Image B>. Every utterance from A or B is preceded with a timestamp closed under brackets.\nFollowing is the dialog history along with image descriptions :\n"
    return prompt



def get_end_prompt(user="A"):
    # here probably just as him to just answer the question 
    prompt = "\nPlease provide the next utterance by answering the question of user "
    if user == "A":
        prompt += "B"
    else:
        prompt += "A"
    prompt += " and by keeping in mind that it's a spoken conversation."
    return prompt


def make_prompt(df, tokenizer, model_id, file, processing):
    prompt = get_start_prompt(processing=processing)
    dialog, answer = get_dialog(df)
    end_prompt = get_end_prompt()
    tokenized = tokenizer([prompt, dialog, end_prompt], return_offsets_mapping=True)
    offsets = tokenized["offset_mapping"]
    input_ids = tokenized["input_ids"]
    if processing is None:
        prompt += dialog
    else: 
        if processing == "windowed":
            prompt_len = len(input_ids[0]) + len(input_ids[1]) + len(input_ids[2])
            split = tokenizer("\n")
            if prompt_len > 3996:
                offset_size = prompt_len - 3996
                indexes = [i for i,t in enumerate(input_ids[1]) if (t == 108) and (i < offset_size)]
                dialog = dialog[offsets[1][indexes[-1]][-1]:]
            #generate summaries if not existing
        if processing == "summary":
            with open("../data/Summary/"+ utils.MODELS[model_id] + "/" + file.split("/")[-1].split(".")[0] + ".txt") as f:
                summary = f.read()
                f.close()
            prompt = get_start_prompt(processing=processing) + summary
    prompt += end_prompt
    return prompt, answer


def load_prompt(files, tokenizer, model_id, processing=None):
    prompts = []
    answers = []
    for f in files : 
        df = pd.read_csv(f)
        prompt, answer = make_prompt(df, tokenizer, model_id, f, processing)
        prompts += [prompt]
        answers += [answer]
    return prompts, answers

def make_summaries(pipe, files, **parameters):
    n = 5
    if "run" in parameters:
        run = parameters.pop("run")
    if "model_id" in parameters:
        model_id = parameters.pop("model_id")
    start = get_start_prompt(processing="noprocessing")
    end = "\nSummarize the conversation without missing any information in less than 200 words."
    files = get_files(Summary, model_id, optional_arg=run.__name__.split('.')[-1])
    print("Generating summaries for ", len(files) ," files")
    for f in tqdm(files) : 
        df = pd.read_csv(f)
        dialog, answer = get_dialog(df)
        split_dialog = dialog.split("\n")[-n-3:]
        last_n = ""
        for d in split_dialog:
            last_n += d + "\n"
        last_n = last_n[:-3]
        summary = pipe([{"role":"user","content":start+dialog+end}], max_new_tokens=300, **parameters)
        print("generated :", summary[0]['generated_text']) #TODO verify all prompts are good
        save(summary[0]['generated_text'] +"\n"+ last_n, run, f, model_id, run.__name__.split(".")[-1])