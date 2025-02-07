# AdaptiveStep: Automatically Dividing Reasoning Step through Model Confidence

  - [2025/1/31]  We have published our paper here.
  
  - [2025/1/31]  We have released our model and data [here](https://huggingface.co/Lux0926).
  
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

To reproduce the BON evaluation results, please go to the `evaluation/math/BON` folder. First, use the `run_all_eval_server.sh` script to specify the PRM and start the PRM server.
```bash
  cd evaluation/math/BON
  conda activate asprm_eval
  bash run_all_eval_server.sh
```



### Code
