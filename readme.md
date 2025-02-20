# AdaptiveStep: Automatically Dividing Reasoning Step through Model Confidence

  - [2025/1/31]  We have released our model and data [here](https://huggingface.co/Lux0926).

  - [2025/2/20]  We have published our paper [here](https://arxiv.org/abs/2502.13943).
  
## Environment Setup
  
  We built the training and evaluation environments based on OpenRLHF and VLLM. We packaged our conda environment using conda-pack and uploaded it to HuggingFace. You can use the following bash command to build the training and evaluation environments.
  
### Train enviroment

First, use the command `which conda` to check the local installation path of Conda ( `{conda_path}` ).
  ```bash
  huggingface-cli download Lux0926/ASPRM-Training-Evaluation-Environment --local-dir {your_save_path}
  mkdir /{conda_path}/asprm_train
  cd {your_save_path}
  tar -xzvf asprm_train.tar.gz -C /{conda_path}/asprm_train
  ```

### Eval enviroment
  ```bash
  mkdir /{conda_path}/asprm_eval
  cd {your_save_path}
  tar -xzvf asprm_eval.tar.gz -C /{conda_path}/asprm_eval
  ```

## Trarning Code

In the `train` folder, we have provided the scripts used for training PRM. To replicate our training process, please run the scripts in the `train` directory after setting up the training environment. We use the [LeetCodeDataset](https://github.com/newfacade/LeetCodeDataset) to perform supervised fine-tuning (SFT) on deepseek-coder-6.7b-base.

#### Example

  ```bash
  cd train/
  conda activate asprm_train
  bash train_ASPRM-M.sh
  ```
#### Script 
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed --module openrlhf.cli.train_prm \
   --save_path ./checkpoint/{your_prm_name} \
   --save_steps 1000 \
   --logging_steps 10 \
   --eval_steps 1000 \
   --train_batch_size 32 \
   --micro_train_batch_size 4 \
   --pretrain {path_to_your_basemodel} \
   --bf16 \
   --max_epochs 1 \
   --max_len 8192 \
   --zero_stage 2 \
   --learning_rate 5e-6 \
   --dataset {path_to_prm_train_dataset} \
   --input_key query \
   --label_key response \
   --flash_attn \
   --packing_samples \
   --wandb_group prm \
   --placeholder_token " ки" \
```
We did not use the `-reward_tokens` parameter, as omitting it typically leads to better performance. 
  
## Evaluation

Please goto the `evaluation/` folder.

### Math
#### BON
To reproduce the BON evaluation results, please go to the `evaluation/math/BON` folder. First, use the `run_all_eval_server.sh` script to specify the PRM and start the PRM server.
```bash
  cd evaluation/math/BON
  conda activate asprm_eval
  bash run_all_eval_server.sh
```
In the `run_all_eval_server.sh`, you can specify the parameters as shown below.
```bash
  bash {run_eval_server_llama.sh/run_eval_server_mistral.sh/run_eval_server_er_prm.sh/run_eval_server_shepherd.sh} {CUDA_VISIBLE_DEVICES} {prm_path} {prm_server_port} &
```

After starting the PRM server, run the `run_eval_code.sh` script to initiate the BON evaluation. You can specify the following parameters within it.
```bash
python eval.py --benchmark_type {llama_data/mistral_data} --bon_size {4/8/16/32/64} --input_data_path {BON_Evaluation_Data_Path}  --reward_port {running_prm_server_port} --eval_type {confidence/hard/random} --dataset_type {gsm8k/math_500} --prm_model_path {prm_model_path} &
```
The complete BON evaluation dataset can be found [here](https://huggingface.co/datasets/Lux0926/ASPRM-BON-Evaluation-Dataset-Math), and the evaluation results will be saved in `math_result.jsonl`. Note that in the BON evaluation dataset, data with the `with_step` suffix is used to evaluate the performance of Shepherd, while data with the `remove_enter` suffix is used to evaluate the performance of ER-PRM.
#### TVD
To reproduce the TVD evaluation results, please go to the `evaluation/math/TVD` folder.
First, run the `run_all_server.sh` script to start the `task_server` and `reward_server`.
```bash
  bash run_all_server.sh
```
You can specify the following parameters.
```bash
# Task model
bash run_task_server.sh {CUDA_VISIBLE_DEVICES} {task_model_path} {task_model_server_port} {task_model_path} &
# PRM model
bash {run_reward_server_mistral.sh/run_reward_server_llama.sh/run_reward_server_er_prm.sh/run_reward_server_shepherd.sh} {CUDA_VISIBLE_DEVICES} {prm_model_path} {prm_model_server_port} &
```
Then, run the `run_tvd.sh` script to initiate the TVD evaluation.

```bash
python tvd.py \
    --temperature {temperature} \
    --bon_size {bon_size：4/8/...} \
    --confidence_threshold {confidence_threshold:0-100} \
    --task_model_path {task_model_path}  \
    --reward_model_path {prm_model_path} \
    --input_data_path {GSM8k_test_idx.jsonl/GSM_symbolic_test_idx.jsonl/math500_test_with_idx.jsonl} \
    --output_data_path {prm_name}_con{confidence_threshold}_{task_model_name}.jsonl \
    --task_port {task_model_server_port} \
    --reward_port {prm_model_server_port} &

wait
```
Upon completion of the run, use `get_score.ipynb` to retrieve the final score for the TVD evaluation.

### Code
The evaluation of Code is similar to that of Math, but all final scores are obtained offline.
#### BON
The complete BON evaluation dataset can be found [here](https://huggingface.co/datasets/Lux0926/ASPRM-BON-Evaluation-Dataset-Code). First, similarly use the `run_all_eval_server.sh` script to specify the PRM and start the PRM server.
```bash
# run_all_eval_server.sh
bash run_eval_server_ds.sh {CUDA_VISIBLE_DEVICES} {prm/orm_path} {prm_server_port} &
```
Also after starting the PRM server, run the `run_eval_code.sh` script to initiate the BON evaluation.
```bash
# run_eval_code.sh
python {eval_lct.py/eval_lcb.py} --benchmark_type {ds_data/ds_data_orm} --bon_size {4/8/16/32/64} --input_data_path {BON_Evaluation_Data_Path} --reward_port {running_prm_server_port} --eval_type {confidence/hard/random} --dataset_type {leetCoTE/LiveCodeBench} --prm_model_path {prm_model_path} &
```
Upon completion, all BON evaluation results will be saved.
#### TVD
Similarly,run the `run_all_server.sh` script to start the `task_server` and `reward_server`.
```bash
# Task model
bash run_task_server.sh {CUDA_VISIBLE_DEVICES} {task_model_path} {task_model_server_port} {task_model_path} &
# PRM model
bash run_reward_server_ds.sh {CUDA_VISIBLE_DEVICES} {prm_model_path} {prm_model_server_port} &
```
Then, to evaluate performance on the LCB dataset, run `run_tvd_lcb.sh`.
```bash
# run_tvd_lcb.sh
(python tvd_lcb.py \
    --temperature {temperature} \
    --bon_size {bon_size：4/8/...} \
    --confidence_threshold {confidence_threshold:0-100} \
    --task_model_path {task_model_path} \
    --reward_model_path {prm_model_path} \
    --input_data_path livecodebench_test_idx.jsonl \
    --output_data_path prm_con{confidence_threshold}_lcb.jsonl \
    --task_port {task_model_server_port} \
    --reward_port {prm_model_server_port} ) &

wait
```
If you want to evaluate performance on the LCT dataset, run `run_tvd_lct.sh`.
```bash
# run_tvd_lct.sh
(python tvd_lct.py \
    --temperature {temperature} \
    --bon_size {bon_size：4/8/...} \
    --confidence_threshold {confidence_threshold:0-100} \
    --task_model_path {task_model_path} \
    --reward_model_path {prm_model_path} \
    --input_data_path leetCoTE_test_idx.jsonl \
    --output_data_path prm_con{confidence_threshold}_lcb.jsonl \
    --task_port {task_model_server_port} \
    --reward_port {prm_model_server_port} ) &

wait
```
Upon completion, all TVD evaluation results will be saved.
#### Get Evaluation Scores
Both BON and TVD evaluations will save the evaluation result files. To obtain the final scores for the evaluation on the LCT dataset, please refer to the following instructions.
```bash
git clone https://github.com/Lux0926/ASPRM_LCT_Eval.git
cd ASPRM_LCT_Eval/
```
To obtain the BON evaluation scores, run `test_leetcode_eval.sh`. To obtain the TVD evaluation scores, run `test_leetcode_tvd.sh`. 
```bash
# test_leetcode_eval.sh
python src/main_eval.py  --model_name "model" \
                --task "LeetCodeTest" \
                --save "sft_old_test" \
                --num_gpus 4 \
                --num_samples 1 \
                --k 1 \
                --temperature 0.0 \
                --num_workers 32 \
                --batch_size 200 \
                --max_tokens 8192 \
                --model_type "Chat" \
                --prompt_type "Instruction" \
                --prompt_prefix "" \
                --prompt_suffix "" \
                --trust_remote_code \
                --input_file {your_evaluation_result_path} \
                --output_file {your_evaluation_result_name}
```

```bash
# test_leetcode_tvd.sh
python src/main_tvd.py  --model_name "model" \
                --task "LeetCodeTest" \
                --save "sft_old_test" \
                --num_gpus 4 \
                --num_samples 1 \
                --k 1 \
                --temperature 0.0 \
                --num_workers 32 \
                --batch_size 200 \
                --max_tokens 8192 \
                --model_type "Chat" \
                --prompt_type "Instruction" \
                --prompt_prefix "" \
                --prompt_suffix "" \
                --trust_remote_code \
                --input_file {your_evaluation_result_path} \
                --output_file {your_evaluation_result_name}
```

Before running these two scripts, simply modify the `--input_file` parameter to specify the file for which you want to obtain the scores. The `--output_file` can be the same as the `--input_file`. No other parameters need to be modified.

To obtain the final scores for the evaluation on the LCB dataset, please follow the official [LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench) GitHub repository. Refer to the "Custom Evaluation" section of the repository.
```bash
python -m lcb_runner.runner.custom_evaluator --custom_output_file {your_evaluation_result_path} --release_version release_v4 &
```
However, please remember that before doing so, you need to use `process_lcb_result.ipynb` to convert the evaluation results into the format required by LiveCodeBench.

## Citation
If you find anything useful for your research, please consider citing.
