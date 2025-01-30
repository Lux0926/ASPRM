import argparse
from typing import Optional
from vllm.transformers_utils.tokenizer import get_tokenizer
import os
from  vllm.entrypoints.llm import SamplingParams,LLM
import numpy as np
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import torch.nn.functional as F
import json
from openai import OpenAI
import os
import sys
import socket
import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from vllm import LLM
import concurrent.futures

from detector import var_detection,confidence_detection
from tqdm import tqdm, trange
import time
from vllm.transformers_utils.tokenizer import get_tokenizer
import json
import concurrent.futures
import threading

            
def read_jsonl(file_path):
    questions = []
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)  
            if 'question' in data:
                questions.append(data['question']) 
                
    return questions

def write_to_jsonl(data, output_file):
    with open(output_file, 'a') as file:
        for entry in data:
            json.dump(entry, file)
            file.write('\n') 

def bon_reward_generate(prompt_list,reward_tokenizer):
    add_tag = ' ки'
    prompt_tag = [s + add_tag for s in prompt_list]
    step_scores_list = []
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_reward,
    )
    models = client.models.list()
    model = models.data[0].id
    responses = client.embeddings.create(
        input=prompt_tag,
        model=model,
    )
    for data in responses.data:
        
        step_scores = torch.tensor(data.embedding).view(-1, 2)[:, 0].item()
        step_scores_list.append(step_scores)

    max_score = max(step_scores_list)
    max_index = step_scores_list.index(max_score)

    torch.cuda.empty_cache()
    return max_score,max_index

def parse_arguments():
    parser = argparse.ArgumentParser(description="Bon Parameters")
    parser.add_argument("--temperature", default=0.0, type=float, required=False, help="Temperature when calculating priors")
    parser.add_argument('--bon_size', default=8, type=int, help='The number of nodes expands at a time')
    parser.add_argument('--confidence_threshold', default=25, type=float, help='Threshold of confidence')
    parser.add_argument('--task_model_path', default=None, type=str, required=True, help='Task model name or model path')
    parser.add_argument('--reward_model_path', default=None, type=str, required=True, help='Reward model name or model path')
    parser.add_argument('--input_data_path', default="test.jsonl", type=str, help='Input datasets path')
    parser.add_argument('--output_data_path', default="result_1112_s2prmtem0con52_boninfer.jsonl", type=str, help='Output datasets path')
    parser.add_argument('--task_port', type=str, required=True)
    parser.add_argument('--reward_port', type=str, required=True)
    return parser.parse_args()

def bon_infer(question,temperature,prompt,task_tokenizer,reward_tokenizer,bon_size,confidence_threshold:float = 10.0):
    prompt = prompt
    temp_prompt = prompt
    question = question
    bon_num = 0
    while True:
        client = OpenAI(api_key=openai_api_key, base_url=openai_api_task)
        models = client.models.list()
        model = models.data[0].id

        outputs = client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=1,
            logprobs=bon_size,
            temperature = temperature
        )
        prompt_len = outputs.usage.total_tokens
        tokens = outputs.choices[0].logprobs.tokens
        token_ids = task_tokenizer.encode(tokens[0])
        if len(token_ids)==1 or prompt_len > 1024:
            break
        logpr = outputs.choices[0].logprobs.token_logprobs[0]
        confidence = confidence_detection(logpr)

        if confidence < confidence_threshold:
            toptoken_dict = outputs.choices[0].logprobs.top_logprobs[0]
            cal_token_list = list(toptoken_dict.keys())
            cal_prompt_list = []
            for token in cal_token_list:
                ori_prompt = prompt+token
                cal_prompt = question+ori_prompt.replace(temp_prompt,"")
                cal_prompt_list.append(cal_prompt)
            max_bon_reward,index = bon_reward_generate(cal_prompt_list,reward_tokenizer=reward_tokenizer)
            bon_token = cal_token_list[index]
            bon_num = bon_num+1
            prompt = prompt + bon_token
            bon_token_ids = task_tokenizer.encode(bon_token)
            if len(bon_token_ids)==1:
                break
            continue
        generated_text = outputs.choices[0].text
        prompt = prompt + generated_text

    torch.cuda.empty_cache()
    return prompt,bon_num

def run_inference_and_write(data):
    idx = data["idx"]
    question = data["question"]
    start_time = time.time()
    text = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{question}\n\n### Response: Let's think step by step."
    answer,bon_num = bon_infer(question=question,prompt=text,bon_size=bon_size,confidence_threshold=confidence_threshold,task_tokenizer=task_tokenizer,reward_tokenizer=reward_tokenizer,temperature=temperature)
    end_time = time.time()
    actual_infer_time = end_time - start_time
    
    output_data = {
            "idx": idx,
            "question":question,
            "answer":answer,
            "bon_num":bon_num,
            "infer_time":actual_infer_time
        }

    with write_lock:
        with open(output_data_path, 'a') as file:
            json.dump(output_data, file)
            file.write('\n')
            
if __name__ == "__main__":
    
    args = parse_arguments()
    
    task_model_path = args.task_model_path
    reward_model_path = args.reward_model_path
    temperature = args.temperature
    bon_size = args.bon_size
    input_data_path = args.input_data_path
    output_data_path = args.output_data_path
    confidence_threshold = args.confidence_threshold
    reward_port = args.reward_port
    task_port = args.task_port

    if os.path.exists(output_data_path):
        os.remove(output_data_path)

    task_tokenizer = get_tokenizer(task_model_path, trust_remote_code=True)
    reward_tokenizer = get_tokenizer(reward_model_path, trust_remote_code=True)
    sorted_output_data_path = "sorted_"+str(output_data_path)
    openai_api_key = "EMPTY"
    openai_api_reward = f"http://localhost:{reward_port}/v1"
    openai_api_task = f"http://localhost:{task_port}/v1"
    data_list = []

    with open(input_data_path, 'r', encoding='utf-8') as file:
        for line in file:
            data_list.append(json.loads(line))

    write_lock = threading.Lock()

    with concurrent.futures.ProcessPoolExecutor(max_workers=48) as executor:
        futures = [executor.submit(run_inference_and_write, data) for data in data_list]
    
        with tqdm(total=len(data_list), desc="Processing prompts") as pbar:
            for future in concurrent.futures.as_completed(futures):
                future.result()  
                pbar.update(1)  

    sorted_data = []
    with open(output_data_path, 'r') as file:
        for line in file:
            sorted_data.append(json.loads(line))
    
    sorted_data.sort(key=lambda x: x['idx'])
    
    with open(sorted_output_data_path, 'w') as file:
        for data in sorted_data:
            json.dump(data, file)
            file.write('\n')

    

    


