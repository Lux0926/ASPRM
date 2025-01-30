from vllm.transformers_utils.tokenizer import get_tokenizer
import re
from openai import OpenAI
import torch
import argparse
import logging
import random
from tqdm import tqdm 
import os
import json

logging.basicConfig(filename='empty_tensor_log.txt', 
                    level=logging.INFO, 
                    format='%(asctime)s - %(message)s')

def parse_arguments():
    parser = argparse.ArgumentParser(description="Bon Parameters")
    parser.add_argument('--bon_size', default=256, type=int, help='The number of nodes expands at a time')
    parser.add_argument('--input_data_path', default="test.jsonl", type=str, help='Input datasets path')
    parser.add_argument('--output_data_path', default="result.jsonl", type=str, help='Output datasets path')
    parser.add_argument('--reward_port', type=str, required=True)
    parser.add_argument('--eval_type', type=str, required=True)
    parser.add_argument('--dataset_type', type=str, required=True)
    parser.add_argument('--benchmark_type', type=str, required=True)
    parser.add_argument('--prm_model_path', type=str, required=True)
    

    return parser.parse_args()

if __name__ == "__main__":
    
    args = parse_arguments()

    bon_size = args.bon_size
    input_data_path = args.input_data_path
    port = args.reward_port
    eval_type = args.eval_type
    dataset_type = args.dataset_type
    prm_model_path = args.prm_model_path
    benchmark_type = args.benchmark_type

    openai_api_key = "EMPTY"
    openai_api_reward = f"http://localhost:{port}/v1"

    
    modelname = os.path.basename(prm_model_path)
    
    output_data_path = f"result_{modelname}_{eval_type}_{dataset_type}_{bon_size}_{benchmark_type}.jsonl"

    if os.path.exists(output_data_path):
        os.remove(output_data_path)

    data_list = []

    with open(input_data_path, 'r', encoding='utf-8') as file:
        for line in file:
            data_list.append(json.loads(line))

    if eval_type == "confidence":
        for data in tqdm(data_list, desc="Processing Confidence Data"): 
            idx = data["idx"]
            question = data["question_content"]
            solution_list = data["code"]

            question_id = data["question_id"]
            math_list = data["code_confidence_list"][:bon_size]
            eval_score_mean = []
            eval_score_max = []
            eval_score_min = []
            for math in math_list:
                if ' ки ' not in math:
                    positions = sorted(random.sample(range(len(math) + 1), 4))
                    for i, pos in enumerate(positions):
                        math = math[:pos + i * 4] + ' ки ' + math[pos + i * 4:]
                try:
                    prompt = question + math
                    client = OpenAI(
                        api_key=openai_api_key,
                        base_url=openai_api_reward,
                    )
                    models = client.models.list()
                    model = models.data[0].id
                    responses = client.embeddings.create(
                        input=prompt,
                        model=model,
                    )
                    for data in responses.data:
                        step_scores = torch.tensor(data.embedding).view(-1, 2)[:, 0]
                        max_value = torch.max(step_scores, dim=0)[0].item()
                        min_value = torch.min(step_scores, dim=0)[0].item()
                        mean_value = torch.mean(step_scores, dim=0).item()
                        eval_score_mean.append(mean_value)
                        eval_score_max.append(max_value)
                        eval_score_min.append(min_value)
                except:
                    continue

            index_mean = eval_score_mean.index(max(eval_score_mean))
            index_min = eval_score_min.index(max(eval_score_min))
            index_max = eval_score_max.index(max(eval_score_max))

            solution_mean = solution_list[index_mean]
            solution_min = solution_list[index_min]
            solution_max = solution_list[index_max]

            output_data = {
                "idx": idx,
                "question": question,
                "solution_mean": solution_mean,
                "solution_max": solution_max,
                "solution_min": solution_min,
                "question_id": question_id
            }
            with open(output_data_path, "a", encoding="utf-8") as f:
                json.dump(output_data, f)
                f.write('\n')

    if eval_type == "random":
        for data in tqdm(data_list, desc="Processing random Data"):  
            idx = data["idx"]
            question = data["question"]
            solution_list = data["pred"]
            answer = data["answer"]
            question_id = data["question_id"]
            math_list = data["code_random_list"][:bon_size]
 
            eval_score_mean = []
            eval_score_max = []
            eval_score_min = []
            for math in math_list:
                if ' ки ' not in math:
                    positions = sorted(random.sample(range(len(math) + 1), 4))
                    for i, pos in enumerate(positions):
                        math = math[:pos + i * 4] + ' ки ' + math[pos + i * 4:]
                try:
                    prompt = question + math
                    # print(prompt)
                    client = OpenAI(
                        api_key=openai_api_key,
                        base_url=openai_api_reward,
                    )
                    models = client.models.list()
                    model = models.data[0].id
                    responses = client.embeddings.create(
                        input=prompt,
                        model=model,
                    )

                    for data in responses.data:
                        step_scores = torch.tensor(data.embedding).view(-1, 2)[:, 0]
                        max_value = torch.max(step_scores, dim=0)[0].item()
                        min_value = torch.min(step_scores, dim=0)[0].item()
                        mean_value = torch.mean(step_scores, dim=0).item()
                        eval_score_mean.append(mean_value)
                        eval_score_max.append(max_value)
                        eval_score_min.append(min_value)
                except:
                    continue

            index_mean = eval_score_mean.index(max(eval_score_mean))
            index_min = eval_score_min.index(max(eval_score_min))
            index_max = eval_score_max.index(max(eval_score_max))

            solution_mean = solution_list[index_mean]
            solution_min = solution_list[index_min]
            solution_max = solution_list[index_max]

            output_data = {
                "idx": idx,
                "question": question,
                "solution_mean": solution_mean,
                "solution_max": solution_max,
                "solution_min": solution_min,
                "answer": answer,
                "question_id": question_id
            }
            with open(output_data_path, "a", encoding="utf-8") as f:
                json.dump(output_data, f)
                f.write('\n')

    if eval_type == "hard":
        for data in tqdm(data_list, desc="Processing hard Data"):  
            idx = data["idx"]
            question = data["question"]
            solution_list = data["pred"]
            answer = data["answer"]
            question_id = data["question_id"]
            math_list = data["code_hard_list"][:bon_size]
            eval_score_mean = []
            eval_score_max = []
            eval_score_min = []
            for math in math_list:
                if ' ки ' not in math:
                    positions = sorted(random.sample(range(len(math) + 1), 4))
                    for i, pos in enumerate(positions):
                        math = math[:pos + i * 4] + ' ки ' + math[pos + i * 4:]
                try:
                    prompt = question + math
                    client = OpenAI(
                        api_key=openai_api_key,
                        base_url=openai_api_reward,
                    )
                    models = client.models.list()
                    model = models.data[0].id
                    responses = client.embeddings.create(
                        input=prompt,
                        model=model,
                    )
                    for data in responses.data:
                        step_scores = torch.tensor(data.embedding).view(-1, 2)[:, 0]
                        max_value = torch.max(step_scores, dim=0)[0].item()
                        min_value = torch.min(step_scores, dim=0)[0].item()
                        mean_value = torch.mean(step_scores, dim=0).item()
                        eval_score_mean.append(mean_value)
                        eval_score_max.append(max_value)
                        eval_score_min.append(min_value)
                except:
                    continue

            index_mean = eval_score_mean.index(max(eval_score_mean))
            index_min = eval_score_min.index(max(eval_score_min))
            index_max = eval_score_max.index(max(eval_score_max))

            solution_mean = solution_list[index_mean]
            solution_min = solution_list[index_min]
            solution_max = solution_list[index_max]

            output_data = {
                "idx": idx,
                "question": question,
                "solution_mean": solution_mean,
                "solution_max": solution_max,
                "solution_min": solution_min,
                "answer": answer,
                "question_id": question_id
            }
            with open(output_data_path, "a", encoding="utf-8") as f:
                json.dump(output_data, f)
                f.write('\n')
