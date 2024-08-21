from transformers import AutoTokenizer, pipeline
import sys
#sys.path.append('../InfiniTransformer/')
from InfiniTransformer.infini_llama.modeling_infini_llama import LlamaForCausalLM
import InfiniTransformer
import torch
import prompt as prompting
from utils import save

def inference(files, **parameters):
    model_path = "../models/llama-3.1-8b-infini-noclm-8192"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    #print(model)
    #print(model.dtype)

    #model = PeftModel.from_pretrained(model, "../InfiniTransformer/models/llama-3.1-8b-infini-noclm-8192")
    #model.load_adapter("../InfiniTransformer/models/llama-3.1-8b-infini-noclm-8192", adapter_name="infini")
    #model.set_adapter("infini")
    
    prompts, answers = prompting.load_prompt(files, tokenizer=tokenizer, model_id=None, processing=None) #TODO connect to real data

    for prompt,file in zip(prompts, files):
        with torch.no_grad():
            generated_text = generate_text_with_stateful_segments(
                model, tokenizer, prompt, max_length=512, temperature=0.8
            )
        save(generated_text[2:], parameters["run"], file, "llama-3.1-8b-infini-noclm-8192")
        
        print("Short-Generated(512) Text: \n", generated_text)

        print("-" * 40)


def generate_text_with_stateful_segments(
    model,
    tokenizer,
    prompt_text,
    max_length=300,
    segment_length=2048,
    temperature=1.0,
):
    # gpu_tracker.track()
    model.eval()
    # gpu_tracker.track()

    # Encode the prompt text
    input_ids = tokenizer.apply_chat_template([{"role":"user", "content":prompt_text}], return_tensors="pt", add_generation_prompt=True)
    #print(tokenizer.decode(input_ids[0]))
    
    original_length = len(input_ids[0])  # Get the original length of the prompt
    print("Original seq len:", original_length)

    # Initialize memory and norm_term
    memory, norm_term = None, None

    # Manage long initial prompts by processing them in segments
    if input_ids.size(1) > segment_length:
        # gpu_tracker.track()
        print("Processing prompt in segments")
        num_segments = input_ids.size(1) // segment_length
        for i in range(num_segments):
            segment = input_ids[:, i * segment_length : (i + 1) * segment_length]
            # gpu_tracker.track()
            outputs = model(
                input_ids=segment.to(model.device), memory=memory, norm_term=norm_term
            )
            # gpu_tracker.track()
            memory = outputs.memory
            norm_term = outputs.norm_term
            # gpu_tracker.track()
        # Handle leftover tokens
        # leftover = input_ids.size(1) % segment_length
        # if leftover > 0:
        #     segment = input_ids[:, -leftover:]
        #     outputs = model(input_ids=segment.to(model.device), memory=memory, norm_term=norm_term)
        #     memory = outputs.memory
        #     norm_term = outputs.norm_term
        print("Prompt/Segments processed, starting generation")
    else:
        print("Short, single-segment prompt, start generation now.")
    # Initialize the generation with the full prompt or the last processed segment
    generated_sequence = input_ids
    print("Target seq len:", original_length + max_length)
    while generated_sequence.size(1) < original_length + max_length:
        #print("generated_sequence.size(1):", generated_sequence.size(1))
        past = None
        # if generated_sequence.size(1) over segment_length, re-compute memory and norm_term
        if generated_sequence.size(1) % segment_length == 0:
            input_segment = generated_sequence[:, -segment_length:]
            # gpu_tracker.track()
            outputs = model(
                input_ids=input_segment.to(model.device),
                memory=memory,
                norm_term=norm_term,
            )
            # gpu_tracker.track()
            # Update memory and norm_term for the next, new segment
            memory = outputs.memory
            norm_term = outputs.norm_term

            # gpu_tracker.track()
            next_token_logits = outputs.logits[:, -1, :]
            scaled_logits = next_token_logits / temperature
            probs = torch.nn.functional.softmax(scaled_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).detach()

            # Append to the generated sequence
            generated_sequence = torch.cat(
                (generated_sequence, next_token.to("cpu")), dim=1
            )
            # gpu_tracker.track()
        else:
            leftover = generated_sequence.size(1) % segment_length
            input_segment = generated_sequence[:, -leftover:]  # Use the last segment

            # gpu_tracker.track()
            outputs = model(
                input_ids=input_segment.to(model.device),
                memory=memory,
                norm_term=norm_term,
                no_memory_update=True,
                use_cache=True,
                past_key_values=past,
            )
            past = outputs.past_key_values
            # gpu_tracker.track()

            # Obtain the last token predictions and sample
            next_token_logits = outputs.logits[:, -1, :]
            scaled_logits = next_token_logits / temperature
            probs = torch.nn.functional.softmax(scaled_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).detach()

            # Append to the generated sequence
            generated_sequence = torch.cat(
                (generated_sequence, next_token.to("cpu")), dim=1
            )
            # gpu_tracker.track()

        # # Break the loop if we reach max_length
        #if generated_sequence.size(1) >= max_length:
        #    break
        
        # Break the loop if model has ended the message
        #print(tokenizer.decode(next_token[0]))#, end='')
        if 128009 in next_token:
            break

    # Decode the generated tokens to text
    generated_text = tokenizer.decode(generated_sequence[0][original_length-1:], skip_special_tokens=True)

    return generated_text.replace(prompt_text, "")