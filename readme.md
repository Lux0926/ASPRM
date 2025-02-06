# AdaptiveStep: Automatically Dividing Reasoning Step through Model Confidence

  - [2025/1/31]  We have published our paper here.
  
  - [2025/1/31]  We have released our model and data [here](https://huggingface.co/Lux0926).
  
## Environment Setup
  
  We built the training and evaluation environments based on OpenRLHF and VLLM. We packaged our conda environment using conda-pack and uploaded it to HuggingFace. You can use the following bash command to build the training and evaluation environments.
  
### Train enviroment

  ```bash
  conda create -n asprm_train python=3.10.0
  conda activate asprm_train
  pip install openrlhf==0.4.5
  ```

### Eval enviroment

## Trarning Code
In the `train` folder, we have provided the scripts used for training PRM. To replicate our training process, please run the scripts in the `train` directory after setting up the training environment.

#### Example

  ```bash
  cd train/
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
