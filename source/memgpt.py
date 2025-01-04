from letta import create_client
from letta.schemas.memory import ChatMemory
from letta import LLMConfig, EmbeddingConfig
from letta.schemas.message import Message,MessageCreate
from letta.schemas.letta_message import SystemMessage, UserMessage
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
        model_wrapper="chatml"
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
        if users[i][-1] == "A":
            users[i] = list(map(lambda x: x.replace('A', 'user').replace('B', 'assistant'), users[i]))
        else :
            users[i] = list(map(lambda x: x.replace('B', 'user').replace('A', 'assistant'), users[i]))
        agent_state = client.create_agent(
            name=d.split("/")[-1] + "_"+f.split("/")[-1].split(".")[0], 
            memory = ChatMemory(
                persona= preprompts[i],
                human="I'm a default user"
            ),
            #TODO PUT HOURS FROM DIALOG  IN MEMGPT MESSAGES METADATA
            initial_message_sequence = [ {"role":users[i][j], "text":m, "user_id":users[i][j]} for j,m in enumerate(messages[i][:-1])]
        )
        
        
        

        #TODO How do we get the answer from the model ?
        response = client.user_message(
            agent_id=agent_state.id,
            message= messages[i][-1], #last message is the query
            include_full_message=True #TODO true or not true ?
        )


print("MEMGPT")
#client.delete_agent("basic_agent")
#print(f"Created agent: {basic_agent}")
all_agents = [c.id for c in client.list_agents()]
print("Number of agents : ", len(all_agents))
print([c.id for c in client.list_agents()])

for a in all_agents:
    client.delete_agent(a)
