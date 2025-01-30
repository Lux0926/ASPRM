#example

python eval.py --benchmark_type mistral_data --bon_size 4 --input_data_path Mistral_GSM8k_testset_bo256.jsonl  --reward_port 8081 --eval_type confidence --dataset_type gsm8k --prm_model_path ASPRM-M &
python eval.py --benchmark_type mistral_data --bon_size 8 --input_data_path Mistral_GSM8k_testset_bo256.jsonl  --reward_port 8081 --eval_type confidence --dataset_type gsm8k --prm_model_path ASPRM-M &
python eval.py --benchmark_type mistral_data --bon_size 16 --input_data_path Mistral_GSM8k_testset_bo256.jsonl  --reward_port 8081 --eval_type confidence --dataset_type gsm8k --prm_model_path ASPRM-M &
python eval.py --benchmark_type mistral_data --bon_size 32 --input_data_path Mistral_GSM8k_testset_bo256.jsonl  --reward_port 8081 --eval_type confidence --dataset_type gsm8k --prm_model_path ASPRM-M &
python eval.py --benchmark_type mistral_data --bon_size 64 --input_data_path Mistral_GSM8k_testset_bo256.jsonl  --reward_port 8081 --eval_type confidence --dataset_type gsm8k --prm_model_path ASPRM-M &

wait
