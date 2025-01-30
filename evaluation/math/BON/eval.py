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
import util
import jsonlines

logging.basicConfig(filename='empty_tensor_log.txt', 
                    level=logging.INFO, 
                    format='%(asctime)s - %(message)s')

def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None

def process_results(completion, answer):
    split_ans = completion.split('The answer is: ')
    if len(split_ans) > 1:
        ans = split_ans[-1]
        extract_ans_temp = ans.split('.\n')[0]
        extract_ans_temp = extract_ans_temp.strip()
        if len(extract_ans_temp)>0 and extract_ans_temp[-1] == '.':
            extract_ans = extract_ans_temp[0:-1]
        else:
            extract_ans = extract_ans_temp
        extract_ans = extract_ans.strip()
        if util.is_equiv(extract_ans, answer):
            return True
        else:
            return False
    else:
        return False

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
    # output_data_path = args.output_data_path
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
            question = data["question"]
            solution_list = data["pred"]
            # answer = data["answer"]
            gt_answer = data["gt_answer"]
            math_list = data["math_confidence_list"][:bon_size]
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
                "gt_answer": gt_answer
            }
            with open(output_data_path, "a", encoding="utf-8") as f:
                json.dump(output_data, f)
                f.write('\n')

    if eval_type == "random":
        for data in tqdm(data_list, desc="Processing random Data"):  
            idx = data["idx"]
            question = data["question"]
            solution_list = data["pred"]
            # answer = data["answer"]
            gt_answer = data["gt_answer"]
            math_list = data["math_random_list"][:bon_size]
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
                "gt_answer": gt_answer
            }
            with open(output_data_path, "a", encoding="utf-8") as f:
                json.dump(output_data, f)
                f.write('\n')

    if eval_type == "hard":
        for data in tqdm(data_list, desc="Processing hard Data"):  
            idx = data["idx"]
            question = data["question"]
            solution_list = data["pred"]
            # answer = data["answer"]
            gt_answer = data["gt_answer"]
            math_list = data["math_hard_list"][:bon_size]
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
                "gt_answer": gt_answer
            }
            with open(output_data_path, "a", encoding="utf-8") as f:
                json.dump(output_data, f)
                f.write('\n')

    eval_data_list = []

    with open(output_data_path, "r", encoding="utf-8") as f2:
        for line in f2:
            eval_data_list.append(json.loads(line))

    if dataset_type == "gsm8k":
        pattern1 = r'The answer is:\s*([+-]?\d*[\.,]?\d+)'
        pattern2 = r'####\s*([+-]?\d*[\.,]?\d+)'

        score_mean = 0
        score_min = 0
        score_max = 0
        for eval_data in eval_data_list:
            solution_mean = eval_data["solution_mean"]
            solution_min = eval_data["solution_min"]
            solution_max = eval_data["solution_max"]
            gt_answer = eval_data["gt_answer"]
            
            match_mean = re.search(pattern1, solution_mean)
            match_min = re.search(pattern1, solution_min)
            match_max = re.search(pattern1, solution_max)

            match_gt = re.search(pattern2,gt_answer)
            
            if match_gt:
                gt_value = match_gt.group(1)

            if match_mean and match_gt:
                solution_mean_value = match_mean.group(1)
                if solution_mean_value == gt_value:
                    score_mean += 1

            if match_min and match_gt:
                solution_min_value = match_min.group(1)
                if solution_min_value == gt_value:
                    score_min += 1

            if match_max and match_gt:
                solution_max_value = match_max.group(1)
                if solution_max_value == gt_value:
                    score_max += 1

        final_score_mean = score_mean/len(eval_data_list)
        final_score_max = score_max/len(eval_data_list)
        final_score_min = score_min/len(eval_data_list)
        print(final_score_mean)
        print(final_score_max)
        print(final_score_min)
        with open("math_result.jsonl","a",encoding="utf-8") as f3:
            result_data= {
                "model":prm_model_path,
                "gsm8k_score_mean":final_score_mean,
                "gsm8k_score_max":final_score_max,
                "gsm8k_score_min":final_score_min,
                "size":bon_size,
                "eval_type":eval_type,
                "dataset_type":dataset_type,
                "benchmark_type":benchmark_type,
                "input_data_path":input_data_path
            }
            json.dump(result_data, f3)
            f3.write('\n')
    
    if dataset_type == "math_500":
        score_mean = 0
        score_min = 0
        score_max = 0
        hendrycks_math_answers = []
        solution_mean_list = []
        solution_max_list = []
        solution_min_list = []
        with open(output_data_path, "r+", encoding="utf8") as f:
            for idx, item in enumerate(jsonlines.Reader(f)):
                gt_answer = item['gt_answer']
                solution_max = item['solution_max']
                solution_mean = item['solution_mean']
                solution_min = item['solution_min']
                temp_ans = remove_boxed(util.last_boxed_only_string(gt_answer))
                hendrycks_math_answers.append(temp_ans)
                solution_mean_list.append(solution_mean)
                solution_max_list.append(solution_max)
                solution_min_list.append(solution_min)

        results_mean = []
        results_max = []
        results_min = []

        for idx, (completion, prompt_answer) in enumerate(zip(solution_mean_list, hendrycks_math_answers)):
            res_mean = process_results(completion, prompt_answer)
            results_mean.append(res_mean)

        for idx, (completion, prompt_answer) in enumerate(zip(solution_max_list, hendrycks_math_answers)):
            res_max = process_results(completion, prompt_answer)
            results_max.append(res_max)

        for idx, (completion, prompt_answer) in enumerate(zip(solution_min_list, hendrycks_math_answers)):
            res_min = process_results(completion, prompt_answer)
            results_min.append(res_min)

        final_score_mean = sum(results_mean)/len(results_mean)
        final_score_min = sum(results_min)/len(results_min)
        final_score_max = sum(results_max)/len(results_max)

        print(final_score_mean)
        print(final_score_max)
        print(final_score_min)

        with open("math_result.jsonl","a",encoding="utf-8") as f3:
            result_data= {
                "model":prm_model_path,
                "math500_score_mean":final_score_mean,
                "math500_score_max":final_score_max,
                "math500_score_min":final_score_min,
                "size":bon_size,
                "eval_type":eval_type,
                "dataset_type":dataset_type,
                "benchmark_type":benchmark_type,
                "input_data_path":input_data_path
            }
            json.dump(result_data, f3)
            f3.write('\n')