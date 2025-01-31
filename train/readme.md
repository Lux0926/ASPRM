## We publicly released our training scripts to enable replication of our PRM training process on our data.

### Example
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
   --dataset {path_to_asprm_train_dataset} \
   --input_key query \
   --label_key response \
   --flash_attn \
   --packing_samples \
   --wandb_group prm \
   --placeholder_token " ки" \
```

### Note:
1. We did not use the `-reward_tokens` parameter, as omitting it typically leads to better performance
