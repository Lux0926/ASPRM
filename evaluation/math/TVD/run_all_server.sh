
# base model
bash run_task_server.sh 0 {task_model_path} 8080 {task_model_path} &

# PRM model
bash run_reward_server_mistral.sh 1 {prm_model_path} 8081 &


