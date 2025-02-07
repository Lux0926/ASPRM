#Example

python eval_lct.py --benchmark_type ds_data --bon_size 4 --input_data_path example_leetCoTE_eval_dataset.jsonl  --reward_port 8080 --eval_type confidence --dataset_type leetCoTE --prm_model_path ASPRM-D  &
python eval_lct.py --benchmark_type ds_data --bon_size 8 --input_data_path example_leetCoTE_eval_dataset.jsonl  --reward_port 8080 --eval_type confidence --dataset_type leetCoTE --prm_model_path ASPRM-D  &
python eval_lct.py --benchmark_type ds_data --bon_size 16 --input_data_path example_leetCoTE_eval_dataset.jsonl  --reward_port 8080 --eval_type confidence --dataset_type leetCoTE --prm_model_path ASPRM-D  &
python eval_lct.py --benchmark_type ds_data --bon_size 32 --input_data_path example_leetCoTE_eval_dataset.jsonl  --reward_port 8080 --eval_type confidence --dataset_type leetCoTE --prm_model_path ASPRM-D  &
python eval_lct.py --benchmark_type ds_data --bon_size 64 --input_data_path example_leetCoTE_eval_dataset.jsonl  --reward_port 8080 --eval_type confidence --dataset_type leetCoTE --prm_model_path ASPRM-D  &


python eval_lct.py --benchmark_type ds_data_orm --bon_size 4 --input_data_path example_leetCoTE_eval_dataset_orm.jsonl  --reward_port 8081 --eval_type confidence --dataset_type leetCoTE --prm_model_path ORM-D  &
python eval_lct.py --benchmark_type ds_data_orm --bon_size 8 --input_data_path example_leetCoTE_eval_dataset_orm.jsonl  --reward_port 8081 --eval_type confidence --dataset_type leetCoTE --prm_model_path ORM-D  &
python eval_lct.py --benchmark_type ds_data_orm --bon_size 16 --input_data_path example_leetCoTE_eval_dataset_orm.jsonl  --reward_port 8081 --eval_type confidence --dataset_type leetCoTE --prm_model_path ORM-D  &
python eval_lct.py --benchmark_type ds_data_orm --bon_size 32 --input_data_path example_leetCoTE_eval_dataset_orm.jsonl  --reward_port 8081 --eval_type confidence --dataset_type leetCoTE --prm_model_path ORM-D  &
python eval_lct.py --benchmark_type ds_data_orm --bon_size 64 --input_data_path example_leetCoTE_eval_dataset_orm.jsonl  --reward_port 8081 --eval_type confidence --dataset_type leetCoTE --prm_model_path ORM-D  &

wait
