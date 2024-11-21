from letta import create_client
from letta.schemas.memory import ChatMemory
from letta import LLMConfig, EmbeddingConfig
from utils import get_files
import prompt as prompting
import os
import memgpt
os.environ['HF_HOME'] = '/data/almanach/user/cdearauj/models/'

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
    for i,f in enumerate(files):
        
        agent = client.create_agent(
            name=d.split("/")[-1] + "_"+f.split("/")[-1].split(".")[0], 
            memory = ChatMemory(
            persona= preprompts[i],
            human="I'm a default user"
        ),
        )
        #TODO PUT MESSAGES IN AGENT.MESSAGES, MODIFY PROMPT WITH USER/AGENT INSTEAD OF USER A/B, each utterance is a message
        #TODO PUT HOURS FROM DIALOG  IN MEMGPT MESSAGES METADATA
        response = client.user_message(
            agent_id=agent.id, 
            message=prompts[i],
            include_full_message=False
        )
        print("1 : ", response)
        
        response = client.user_message(
            agent_id=agent.id, 
            message= queries[-1], 
            include_full_message=False
        )
        print("2 : ", response)
        response = client.user_message(
            agent_id=agent.id, 
            message= "/memory", 
            include_full_message=False
        )
        print("3 : ", response)

        
    #print([c.id for c in client.list_agents()])

print("MEMGPT")
#client.delete_agent("basic_agent")
#print(f"Created agent: {basic_agent}")
all_agents = [c.id for c in client.list_agents()]
print("Number of agents : ", len(all_agents))
print([c.id for c in client.list_agents()])

for a in all_agents:
    client.delete_agent(a)
                        
