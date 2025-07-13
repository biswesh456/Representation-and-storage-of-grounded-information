from letta import create_client
from letta.schemas.memory import ChatMemory
from letta import LLMConfig, EmbeddingConfig
from letta.schemas.message import Message,MessageCreate, Message
from letta.schemas.letta_message import SystemMessage, UserMessage
from datetime import datetime 
from utils import get_files
import prompt as prompting
import os
import memgpt
from time import sleep
os.environ['HF_HOME'] = '/data/almanach/user/cdearauj/models/'
#os.environ['HF_HOME'] = '/data/clustext/user/bmohapat/huggingface/'

base_url = "http://localhost:8283"

from letta import EmbeddingConfig

#odel_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
model_id = "mistralai/Mistral-Nemo-Instruct-2407"

#model_id = "google/gemma-2-27b-it"
run = "memgpt"


client = create_client(base_url=base_url)
# embedding model specification
client.set_default_embedding_config( 
embedding_config = EmbeddingConfig(
    embedding_endpoint_type="hugging-face",
    embedding_endpoint="http://127.0.0.1:8080",
    embedding_model="BAAI/bge-large-en-v1.5",
    embedding_dim=1024,
    embedding_chunk_size=300
    )
)
client.set_default_llm_config(
    LLMConfig(
        model=model_id,
        model_endpoint_type="vllm",
        model_endpoint="http://localhost:8000/v1/",
        context_window=8000,
        model_wrapper="chatml-hints-grammar"
    )
)

all_agents = [c.id for c in client.list_agents()]
for a in all_agents:
    client.delete_agent(a)

# loop over files
directories = [ f.path for f in os.scandir("../data/meetup_target/") if f.is_dir() ]
for d in directories:
    files = get_files(memgpt, model_id, CoT=False, dataset_name=d.split("/")[-1]) 
    preprompts, prompts, queries = prompting.load_prompt(files, tokenizer=None, model_id=None, processing="memgpt", CoT=False, dataset_name=None)
    messages, times, users, answer, answer_user, answer_time = prompting.load_prompt(files, tokenizer=None, model_id=None, processing="memgpt-messages", CoT=False, dataset_name=None)
    for i,f in enumerate(files):
        # take "-1" times coming from added image descriptions and interpolate to get new time not breaking code
        newtime = []
        for j,t in enumerate(times[i]):
            if t == "-1":
                if j == 0:
                    newtime += [datetime.strptime(times[i][j+1], "%H:%M:%S")]
                    continue
                
                if j == len(times[i])-1:
                    newtime += [datetime.strptime(times[i][j-1], "%H:%M:%S")]
                    continue
                newtime += [datetime.strptime(times[i][j-1], "%H:%M:%S") + ((datetime.strptime(times[i][j+1], "%H:%M:%S") - datetime.strptime(times[i][j-1], "%H:%M:%S")) / 2)]
            else:
                newtime += [datetime.strptime(t, "%H:%M:%S")]

        if users[i][-1] == "A":
            users[i] = list(map(lambda x: x.replace('A', 'user').replace('B', 'assistant'), users[i]))
        else :
            users[i] = list(map(lambda x: x.replace('B', 'user').replace('A', 'assistant'), users[i]))
        agent_state = client.create_agent(
            name=d.split("/")[-1] + "_"+f.split("/")[-1].split(".")[0], 
            memory = ChatMemory(
                persona= preprompts[i] + "and I always remember to 'send_message' to chat with my user.",
                human="I'm a default user."
            ),
            system="Don't forget the params field in the JSON response",#TODO maybe better instructions
            initial_message_sequence = [ {"role":users[i][j], "text":m, "user_id":users[i][j], "created_at": newtime[j]} for j,m in enumerate(messages[i][:-1])]
        )
        
        
        #TODO send_message -> tool_calls

        #TODO How do we get the answer from the model ?
        try : 
            response = client.user_message(
                agent_id=agent_state.id,
                message= messages[i][-1], #last message is the query
                include_full_message=True #TODO true or not true ?
            )
        except:
            #TODO good handling of error
            print("error")

print("MEMGPT")
#client.delete_agent("basic_agent")
#print(f"Created agent: {basic_agent}")
all_agents = [c.id for c in client.list_agents()]
print("Number of agents : ", len(all_agents))
print([c.id for c in client.list_agents()])

for a in all_agents:
    client.delete_agent(a)
