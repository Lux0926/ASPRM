#Example

(python tvd_lcb.py \
    --temperature 0.0 \
    --bon_size 8 \
    --confidence_threshold 28 \
    --task_model_path {task_model_path} \
    --reward_model_path {prm_path} \
    --input_data_path livecodebench_test_idx.jsonl \
    --output_data_path prm_con28_lcb.jsonl \
    --task_port 8080 \
    --reward_port 8081 ) &

wait
