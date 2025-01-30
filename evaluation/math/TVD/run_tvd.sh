python bon_infer.py \
    --temperature 0.0 \
    --bon_size 8 \
    --confidence_threshold 55 \
    --task_model_path {task_model_path}  \
    --reward_model_path {prm_model_path} \
    --input_data_path GSM8k_test_idx.jsonl \
    --output_data_path {prm_name}_con55_{task_model_name}.jsonl \
    --task_port 8080 \
    --reward_port 8081 &

wait
