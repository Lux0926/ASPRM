
# s2prm_datamistralv3_modelllama_bz256_lr2e6_epo1_no_con
# s2prm_datamistralv3_modelmistral_bz256_lr1e6_epo1_no_con

python bon_infer.py \
    --temperature 0.0 \
    --bon_size 8 \
    --confidence_threshold 55 \
    --task_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/models/metamath_llama31_8b  \
    --reward_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/S2PRM/OpenRLHF/examples/scripts/checkpoint/s2prm_datamistralv3_modelmistral_bz256_lr1e6_epo1_no_con \
    --input_data_path GSM8k_test_idx.jsonl \
    --output_data_path bon_infer_s2prm_datamistralv3_modelmistral_bz256_lr1e6_epo1_no_con_con55_gsm_llama.jsonl \
    --task_port 8080 \
    --reward_port 8081 &

python bon_infer.py \
    --temperature 0.0 \
    --bon_size 8 \
    --confidence_threshold 55 \
    --task_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/models/metamath_llama31_8b  \
    --reward_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/S2PRM/OpenRLHF/examples/scripts/checkpoint/s2prm_datamistralv3_modelllama_bz256_lr2e6_epo1_no_con \
    --input_data_path GSM8k_test_idx.jsonl \
    --output_data_path bon_infer_s2prm_datamistralv3_modelllama_bz256_lr2e6_epo1_no_con_con55_gsm_llama.jsonl \
    --task_port 8090 \
    --reward_port 8083 

wait

python bon_infer.py \
    --temperature 0.0 \
    --bon_size 8 \
    --confidence_threshold 52 \
    --task_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/models/metamath_llama31_8b  \
    --reward_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/S2PRM/OpenRLHF/examples/scripts/checkpoint/s2prm_datamistralv3_modelmistral_bz256_lr1e6_epo1_no_con \
    --input_data_path GSM8k_test_idx.jsonl \
    --output_data_path bon_infer_s2prm_datamistralv3_modelmistral_bz256_lr1e6_epo1_no_con_con52_gsm_llama.jsonl \
    --task_port 8080 \
    --reward_port 8081 &

python bon_infer.py \
    --temperature 0.0 \
    --bon_size 8 \
    --confidence_threshold 52 \
    --task_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/models/metamath_llama31_8b  \
    --reward_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/S2PRM/OpenRLHF/examples/scripts/checkpoint/s2prm_datamistralv3_modelllama_bz256_lr2e6_epo1_no_con \
    --input_data_path GSM8k_test_idx.jsonl \
    --output_data_path bon_infer_s2prm_datamistralv3_modelllama_bz256_lr2e6_epo1_no_con_con52_gsm_llama.jsonl \
    --task_port 8090 \
    --reward_port 8083 

wait

python bon_infer.py \
    --temperature 0.0 \
    --bon_size 8 \
    --confidence_threshold 50 \
    --task_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/models/metamath_llama31_8b  \
    --reward_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/S2PRM/OpenRLHF/examples/scripts/checkpoint/s2prm_datamistralv3_modelmistral_bz256_lr1e6_epo1_no_con \
    --input_data_path GSM8k_test_idx.jsonl \
    --output_data_path bon_infer_s2prm_datamistralv3_modelmistral_bz256_lr1e6_epo1_no_con_con50_gsm_llama.jsonl \
    --task_port 8080 \
    --reward_port 8081 &

python bon_infer.py \
    --temperature 0.0 \
    --bon_size 8 \
    --confidence_threshold 50 \
    --task_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/models/metamath_llama31_8b  \
    --reward_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/S2PRM/OpenRLHF/examples/scripts/checkpoint/s2prm_datamistralv3_modelllama_bz256_lr2e6_epo1_no_con \
    --input_data_path GSM8k_test_idx.jsonl \
    --output_data_path bon_infer_s2prm_datamistralv3_modelllama_bz256_lr2e6_epo1_no_con_con50_gsm_llama.jsonl \
    --task_port 8090 \
    --reward_port 8083 

wait


python bon_infer.py \
    --temperature 0.0 \
    --bon_size 8 \
    --confidence_threshold 40 \
    --task_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/models/metamath_llama31_8b  \
    --reward_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/S2PRM/OpenRLHF/examples/scripts/checkpoint/s2prm_datamistralv3_modelmistral_bz256_lr1e6_epo1_no_con \
    --input_data_path math500_test_with_idx.jsonl \
    --output_data_path bon_infer_s2prm_datamistralv3_modelmistral_bz256_lr1e6_epo1_no_con_con40_math500_llama.jsonl \
    --task_port 8080 \
    --reward_port 8081 &

python bon_infer.py \
    --temperature 0.0 \
    --bon_size 8 \
    --confidence_threshold 40 \
    --task_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/models/metamath_llama31_8b  \
    --reward_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/S2PRM/OpenRLHF/examples/scripts/checkpoint/s2prm_datamistralv3_modelllama_bz256_lr2e6_epo1_no_con \
    --input_data_path math500_test_with_idx.jsonl \
    --output_data_path bon_infer_s2prm_datamistralv3_modelllama_bz256_lr2e6_epo1_no_con_con40_math500_llama.jsonl \
    --task_port 8090 \
    --reward_port 8083 



wait

python bon_infer.py \
    --temperature 0.0 \
    --bon_size 8 \
    --confidence_threshold 45 \
    --task_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/models/metamath_llama31_8b  \
    --reward_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/S2PRM/OpenRLHF/examples/scripts/checkpoint/s2prm_datamistralv3_modelmistral_bz256_lr1e6_epo1_no_con \
    --input_data_path math500_test_with_idx.jsonl \
    --output_data_path bon_infer_s2prm_datamistralv3_modelmistral_bz256_lr1e6_epo1_no_con_con45_math500_llama.jsonl \
    --task_port 8080 \
    --reward_port 8081 &

python bon_infer.py \
    --temperature 0.0 \
    --bon_size 8 \
    --confidence_threshold 45 \
    --task_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/models/metamath_llama31_8b  \
    --reward_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/S2PRM/OpenRLHF/examples/scripts/checkpoint/s2prm_datamistralv3_modelllama_bz256_lr2e6_epo1_no_con \
    --input_data_path math500_test_with_idx.jsonl \
    --output_data_path bon_infer_s2prm_datamistralv3_modelllama_bz256_lr2e6_epo1_no_con_con45_math500_llama.jsonl \
    --task_port 8090 \
    --reward_port 8083 



wait

python bon_infer.py \
    --temperature 0.0 \
    --bon_size 8 \
    --confidence_threshold 42 \
    --task_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/models/metamath_llama31_8b  \
    --reward_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/S2PRM/OpenRLHF/examples/scripts/checkpoint/s2prm_datamistralv3_modelmistral_bz256_lr1e6_epo1_no_con \
    --input_data_path math500_test_with_idx.jsonl \
    --output_data_path bon_infer_s2prm_datamistralv3_modelmistral_bz256_lr1e6_epo1_no_con_con42_math500_llama.jsonl \
    --task_port 8080 \
    --reward_port 8081 &

python bon_infer.py \
    --temperature 0.0 \
    --bon_size 8 \
    --confidence_threshold 42 \
    --task_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/models/metamath_llama31_8b  \
    --reward_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/S2PRM/OpenRLHF/examples/scripts/checkpoint/s2prm_datamistralv3_modelllama_bz256_lr2e6_epo1_no_con \
    --input_data_path math500_test_with_idx.jsonl \
    --output_data_path bon_infer_s2prm_datamistralv3_modelllama_bz256_lr2e6_epo1_no_con_con42_math500_llama.jsonl \
    --task_port 8090 \
    --reward_port 8083 



wait


# python bon_infer.py \
#     --temperature 0.0 \
#     --bon_size 8 \
#     --confidence_threshold 35 \
#     --task_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/models/metamath_llama31_8b  \
#     --reward_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/S2PRM/OpenRLHF/examples/scripts/checkpoint/s2prm_datamistralv3_modelmistral_bz256_lr1e6_epo1_no_con \
#     --input_data_path GSM8k_test_idx.jsonl \
#     --output_data_path bon_infer_s2prm_datamistral_modelmistral_bz256_lr1e6_epo1_no_con_con35_math500.jsonl \
#     --task_port 8080 \
#     --reward_port 8081 &

# python bon_infer.py \
#     --temperature 0.0 \
#     --bon_size 8 \
#     --confidence_threshold 35 \
#     --task_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/models/metamath_llama31_8b  \
#     --reward_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/S2PRM/OpenRLHF/examples/scripts/checkpoint/s2prm_datamistralv3_modelllama_bz256_lr2e6_epo1_no_con \
#     --input_data_path GSM8k_test_idx.jsonl \
#     --output_data_path bon_infer_s2prm_datamistralv3_modelllama_bz256_lr2e6_epo1_no_con_con35_math500.jsonl \
#     --task_port 8090 \
#     --reward_port 8083 



# wait

# python bon_infer.py \
#     --temperature 0.0 \
#     --bon_size 8 \
#     --confidence_threshold 40 \
#     --task_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/models/metamath_llama31_8b  \
#     --reward_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/S2PRM/OpenRLHF/examples/scripts/checkpoint/s2prm_datamistralv3_modelmistral_bz256_lr1e6_epo1_no_con \
#     --input_data_path GSM8k_test_idx.jsonl \
#     --output_data_path bon_infer_s2prm_datamistral_modelmistral_bz256_lr1e6_epo1_no_con_con40_math500.jsonl \
#     --task_port 8080 \
#     --reward_port 8081 &

# python bon_infer.py \
#     --temperature 0.0 \
#     --bon_size 8 \
#     --confidence_threshold 40 \
#     --task_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/models/metamath_llama31_8b  \
#     --reward_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/S2PRM/OpenRLHF/examples/scripts/checkpoint/s2prm_datamistralv3_modelllama_bz256_lr2e6_epo1_no_con \
#     --input_data_path GSM8k_test_idx.jsonl \
#     --output_data_path bon_infer_s2prm_datamistralv3_modelllama_bz256_lr2e6_epo1_no_con_con40_math500.jsonl \
#     --task_port 8090 \
#     --reward_port 8083 



# wait

# python bon_infer.py \
#     --temperature 0.0 \
#     --bon_size 8 \
#     --confidence_threshold 45 \
#     --task_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/models/metamath_llama31_8b  \
#     --reward_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/S2PRM/OpenRLHF/examples/scripts/checkpoint/s2prm_datamistralv3_modelmistral_bz256_lr1e6_epo1_no_con \
#     --input_data_path GSM8k_test_idx.jsonl \
#     --output_data_path bon_infer_s2prm_datamistral_modelmistral_bz256_lr1e6_epo1_no_con_con45_math500.jsonl \
#     --task_port 8080 \
#     --reward_port 8081 &

# python bon_infer.py \
#     --temperature 0.0 \
#     --bon_size 8 \
#     --confidence_threshold 45 \
#     --task_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/models/metamath_llama31_8b  \
#     --reward_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/S2PRM/OpenRLHF/examples/scripts/checkpoint/s2prm_datamistralv3_modelllama_bz256_lr2e6_epo1_no_con \
#     --input_data_path GSM8k_test_idx.jsonl \
#     --output_data_path bon_infer_s2prm_datamistralv3_modelllama_bz256_lr2e6_epo1_no_con_con45_math500.jsonl \
#     --task_port 8090 \
#     --reward_port 8083 



# wait



# python bon_infer.py \
#     --temperature 0.0 \
#     --bon_size 8 \
#     --confidence_threshold 50 \
#     --task_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/models/metamath_llama31_8b  \
#     --reward_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/S2PRM/OpenRLHF/examples/scripts/checkpoint/s2prm_datamistralv3_modelmistral_bz256_lr1e6_epo1_no_con \
#     --input_data_path GSM8k_test_idx.jsonl \
#     --output_data_path bon_infer_s2prm_datamistral_modelmistral_bz256_lr1e6_epo1_no_con_con50_math500.jsonl \
#     --task_port 8080 \
#     --reward_port 8081 &

# python bon_infer.py \
#     --temperature 0.0 \
#     --bon_size 8 \
#     --confidence_threshold 50 \
#     --task_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/models/metamath_llama31_8b  \
#     --reward_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/S2PRM/OpenRLHF/examples/scripts/checkpoint/s2prm_datamistralv3_modelllama_bz256_lr2e6_epo1_no_con \
#     --input_data_path GSM8k_test_idx.jsonl \
#     --output_data_path bon_infer_s2prm_datamistralv3_modelllama_bz256_lr2e6_epo1_no_con_con50_math500.jsonl \
#     --task_port 8090 \
#     --reward_port 8083 



# wait


# # python bon_infer.py \
# #     --temperature 0.0 \
# #     --bon_size 8 \
# #     --confidence_threshold 55 \
# #     --task_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/models/metamath_llama31_8b  \
# #     --reward_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/S2PRM/OpenRLHF/examples/scripts/checkpoint/s2prm_datamistralv3_modelmistral_bz256_lr1e6_epo1_no_con \
# #     --input_data_path GSM8k_test_idx.jsonl \
# #     --output_data_path bon_infer_s2prm_datamistral_modelmistral_bz256_lr1e6_epo1_no_con_con55_math500.jsonl \
# #     --task_port 8080 \
# #     --reward_port 8081 &

# # python bon_infer.py \
# #     --temperature 0.0 \
# #     --bon_size 8 \
# #     --confidence_threshold 55 \
# #     --task_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/models/metamath_llama31_8b  \
# #     --reward_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/S2PRM/OpenRLHF/examples/scripts/checkpoint/s2prm_datamistralv3_modelllama_bz256_lr2e6_epo1_no_con \
# #     --input_data_path GSM8k_test_idx.jsonl \
# #     --output_data_path bon_infer_s2prm_datamistralv3_modelllama_bz256_lr2e6_epo1_no_con_con55_math500.jsonl \
# #     --task_port 8090 \
# #     --reward_port 8083 



# # wait


# python bon_infer.py \
#     --temperature 0.0 \
#     --bon_size 8 \
#     --confidence_threshold 0 \
#     --task_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/models/metamath_llama31_8b  \
#     --reward_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/S2PRM/OpenRLHF/examples/scripts/checkpoint/s2prm_datamistralv3_modelmistral_bz256_lr1e6_epo1_no_con \
#     --input_data_path GSM8k_test_idx.jsonl \
#     --output_data_path bon_infer_s2prm_datamistral_modelmistral_bz256_lr1e6_epo1_no_con_con0_math500.jsonl \
#     --task_port 8080 \
#     --reward_port 8081 &

# python bon_infer.py \
#     --temperature 0.0 \
#     --bon_size 8 \
#     --confidence_threshold 56 \
#     --task_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/models/metamath_llama31_8b  \
#     --reward_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/S2PRM/OpenRLHF/examples/scripts/checkpoint/s2prm_datamistral_modelmistral_bz256_lr1e6_epo1_no_con \
#     --input_data_path GSM8k_test_idx.jsonl \
#     --output_data_path bon_infer_s2prm_datamistral_modelmistral_bz256_lr1e6_epo1_no_con_con56_gsm8k.jsonl \
#     --task_port 8080 \
#     --reward_port 8081 &

# python bon_infer.py \
#     --temperature 0.0 \
#     --bon_size 8 \
#     --confidence_threshold 55 \
#     --task_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/models/metamath_llama31_8b  \
#     --reward_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/S2PRM/OpenRLHF/examples/scripts/checkpoint/s2prm_datamistral_modelmistral_bz256_lr1e6_epo1_no_con \
#     --input_data_path GSM8k_test_idx.jsonl \
#     --output_data_path bon_infer_s2prm_datamistral_modelmistral_bz256_lr1e6_epo1_no_con_con55_gsm8kv2.jsonl \
#     --task_port 8080 \
#     --reward_port 8081 &

# wait


# python bon_infer_tempelete.py \
#     --temperature 0.0 \
#     --bon_size 8 \
#     --confidence_threshold 55 \
#     --task_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/models/metamath_llama31_8b  \
#     --reward_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/S2PRM/OpenRLHF/examples/scripts/checkpoint/s2prm_datamistral_modelmistral_bz256_lr1e6_epo1_no_con \
#     --input_data_path GSM8k_test_idx.jsonl \
#     --output_data_path s2prm_datamistral_modelmistral_bz256_lr1e6_epo1_no_con_con55_gsm8k_basellama.jsonl \
#     --task_port 8080 \
#     --reward_port 8081 &

# python bon_infer.py \
#     --temperature 0.0 \
#     --bon_size 8 \
#     --confidence_threshold 56 \
#     --task_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/models/metamath_llama31_8b  \
#     --reward_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/S2PRM/OpenRLHF/examples/scripts/checkpoint/s2prm_datamistral_modelmistral_bz256_lr1e6_epo1_no_con \
#     --input_data_path GSM8k_test_idx.jsonl \
#     --output_data_path s2prm_datamistral_modelmistral_bz256_lr1e6_epo1_no_con_con56_gsm8k_basellama.jsonl \
#     --task_port 8080 \
#     --reward_port 8081 &


# python bon_infer.py \
#     --temperature 0.0 \
#     --bon_size 8 \
#     --confidence_threshold 57 \
#     --task_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/models/metamath_llama31_8b  \
#     --reward_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/S2PRM/OpenRLHF/examples/scripts/checkpoint/s2prm_datamistral_modelmistral_bz256_lr1e6_epo1_no_con \
#     --input_data_path GSM8k_test_idx.jsonl \
#     --output_data_path s2prm_datamistral_modelmistral_bz256_lr1e6_epo1_no_con_con57_gsm8k_basellama.jsonl \
#     --task_port 8080 \
#     --reward_port 8081 &

# wait

# python bon_infer.py \
#     --temperature 0.0 \
#     --bon_size 8 \
#     --confidence_threshold 54 \
#     --task_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/models/metamath_llama31_8b  \
#     --reward_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/S2PRM/OpenRLHF/examples/scripts/checkpoint/s2prm_datamistral_modelllama_bz256_lr2e6_epo1_no_con \
#     --input_data_path GSM8k_test_idx.jsonl \
#     --output_data_path s2prm_datamistral_modelllama_bz256_lr2e6_epo1_no_con_con54_gsm8k_basellama.jsonl \
#     --task_port 8080 \
#     --reward_port 8081 &

# python bon_infer_tempelete.py \
#     --temperature 0.0 \
#     --bon_size 8 \
#     --confidence_threshold 54 \
#     --task_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/models/metamath_llama31_8b  \
#     --reward_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/S2PRM/OpenRLHF/examples/scripts/checkpoint/s2prm_datamistral_modelllama_bz256_lr2e6_epo1_no_con \
#     --input_data_path GSM8k_test_idx.jsonl \
#     --output_data_path s2prm_datamistral_modelllama_bz256_lr2e6_epo1_no_con_con54_gsm8k_tempelete_basellama.jsonl \
#     --task_port 8080 \
#     --reward_port 8081 &

# python bon_infer_tempelete.py \
#     --temperature 0.0 \
#     --bon_size 8 \
#     --confidence_threshold 54 \
#     --task_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/models/metamath_llama31_8b  \
#     --reward_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/models/shepherd_prm \
#     --input_data_path GSM8k_test_idx.jsonl \
#     --output_data_path shepherd_prm_modelmistral_con54  _gsm8k_tempelete_basellama.jsonl \
#     --task_port 8090 \
#     --reward_port 8083 &

# python bon_infer.py \
#     --temperature 0.0 \
#     --bon_size 8 \
#     --confidence_threshold 54 \
#     --task_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/models/metamath_llama31_8b  \
#     --reward_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/models/shepherd_prm \
#     --input_data_path GSM8k_test_idx.jsonl \
#     --output_data_path shepherd_prm_modelmistral_con54_gsm8k_basellama.jsonl \
#     --task_port 8090 \
#     --reward_port 8083 &

# wait

# python bon_infer.py \
#     --temperature 0.0 \
#     --bon_size 8 \
#     --confidence_threshold 54 \
#     --task_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/models/metamath_llama31_8b  \
#     --reward_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/S2PRM/OpenRLHF/examples/scripts/checkpoint/s2prm_datamistral_modelmistral_bz256_lr1e6_epo1_no_con \
#     --input_data_path GSM8k_test_idx.jsonl \
#     --output_data_path bon_infer_s2prm_datamistral_modelmistral_bz256_lr1e6_epo1_no_con_con54_gsm8k.jsonl \
#     --task_port 8080 \
#     --reward_port 8081 &

# python bon_infer_tempelete.py \
#     --temperature 0.0 \
#     --bon_size 8 \
#     --confidence_threshold 54 \
#     --task_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/models/metamath_llama31_8b  \
#     --reward_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/S2PRM/OpenRLHF/examples/scripts/checkpoint/s2prm_datamistral_modelmistral_bz256_lr1e6_epo1_no_con \
#     --input_data_path GSM8k_test_idx.jsonl \
#     --output_data_path bon_infer_s2prm_datamistral_modelmistral_bz256_lr1e6_epo1_no_con_con54_gsm8k_tempelete.jsonl \
#     --task_port 8080 \
#     --reward_port 8081 &

# wait

# python bon_infer.py \
#     --temperature 0.0 \
#     --bon_size 8 \
#     --confidence_threshold 52 \
#     --task_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/models/metamath_llama31_8b  \
#     --reward_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/S2PRM/OpenRLHF/examples/scripts/checkpoint/s2prm_datamistral_modelmistral_bz256_lr1e6_epo1_no_con \
#     --input_data_path GSM8k_test_idx.jsonl \
#     --output_data_path bon_infer_s2prm_datamistral_modelmistral_bz256_lr1e6_epo1_no_con_con52_gsm8k.jsonl \
#     --task_port 8080 \
#     --reward_port 8081 &

# python bon_infer_tempelete.py \
#     --temperature 0.0 \
#     --bon_size 8 \
#     --confidence_threshold 52 \
#     --task_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/models/metamath_llama31_8b  \
#     --reward_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/S2PRM/OpenRLHF/examples/scripts/checkpoint/s2prm_datamistral_modelmistral_bz256_lr1e6_epo1_no_con \
#     --input_data_path GSM8k_test_idx.jsonl \
#     --output_data_path bon_infer_s2prm_datamistral_modelmistral_bz256_lr1e6_epo1_no_con_con52_gsm8k_tempelete.jsonl \
#     --task_port 8080 \
#     --reward_port 8081 &

# wait

# python bon_infer.py \
#     --temperature 0.0 \
#     --bon_size 8 \
#     --confidence_threshold 15 \
#     --task_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/models/metamath_llama31_8b  \
#     --reward_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/S2PRM/OpenRLHF/examples/scripts/checkpoint/s2prm_datamistral_modelmistral_bz256_lr1e6_epo1_no_con \
#     --input_data_path GSM8k_test_idx.jsonl \
#     --output_data_path bon_infer_s2prm_datamistral_modelmistral_bz256_lr1e6_epo1_no_con_con15_gsm8k.jsonl \
#     --task_port 8080 \
#     --reward_port 8081 &

# python bon_infer_tempelete.py \
#     --temperature 0.0 \
#     --bon_size 8 \
#     --confidence_threshold 15 \
#     --task_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/models/metamath_llama31_8b  \
#     --reward_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/S2PRM/OpenRLHF/examples/scripts/checkpoint/s2prm_datamistral_modelmistral_bz256_lr1e6_epo1_no_con \
#     --input_data_path GSM8k_test_idx.jsonl \
#     --output_data_path bon_infer_s2prm_datamistral_modelmistral_bz256_lr1e6_epo1_no_con_con15_gsm8k_tempelete.jsonl \
#     --task_port 8080 \
#     --reward_port 8081 &

# wait

# python bon_infer.py \
#     --temperature 0.0 \
#     --bon_size 8 \
#     --confidence_threshold 25 \
#     --task_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/models/metamath_llama31_8b  \
#     --reward_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/S2PRM/OpenRLHF/examples/scripts/checkpoint/s2prm_datamistral_modelmistral_bz256_lr1e6_epo1_no_con \
#     --input_data_path GSM8k_test_idx.jsonl \
#     --output_data_path bon_infer_s2prm_datamistral_modelmistral_bz256_lr1e6_epo1_no_con_con25_gsm8k.jsonl \
#     --task_port 8080 \
#     --reward_port 8081 &

# python bon_infer_tempelete.py \
#     --temperature 0.0 \
#     --bon_size 8 \
#     --confidence_threshold 25 \
#     --task_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/models/metamath_llama31_8b  \
#     --reward_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/S2PRM/OpenRLHF/examples/scripts/checkpoint/s2prm_datamistral_modelmistral_bz256_lr1e6_epo1_no_con \
#     --input_data_path GSM8k_test_idx.jsonl \
#     --output_data_path bon_infer_s2prm_datamistral_modelmistral_bz256_lr1e6_epo1_no_con_con25_gsm8k_tempelete.jsonl \
#     --task_port 8080 \
#     --reward_port 8081 &

# wait

# python bon_infer.py \
#     --temperature 0.0 \
#     --bon_size 8 \
#     --confidence_threshold 65 \
#     --task_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/models/metamath_llama31_8b  \
#     --reward_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/S2PRM/OpenRLHF/examples/scripts/checkpoint/s2prm_datamistral_modelmistral_bz256_lr1e6_epo1_no_con \
#     --input_data_path GSM8k_test_idx.jsonl \
#     --output_data_path bon_infer_s2prm_datamistral_modelmistral_bz256_lr1e6_epo1_no_con_con65_gsm8k.jsonl \
#     --task_port 8080 \
#     --reward_port 8081 &

# python bon_infer_tempelete.py \
#     --temperature 0.0 \
#     --bon_size 8 \
#     --confidence_threshold 65 \
#     --task_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/models/metamath_llama31_8b  \
#     --reward_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/S2PRM/OpenRLHF/examples/scripts/checkpoint/s2prm_datamistral_modelmistral_bz256_lr1e6_epo1_no_con \
#     --input_data_path GSM8k_test_idx.jsonl \
#     --output_data_path bon_infer_s2prm_datamistral_modelmistral_bz256_lr1e6_epo1_no_con_con65_gsm8k_tempelete.jsonl \
#     --task_port 8080 \
#     --reward_port 8081 &

# wait

# python bon_infer.py \
#     --temperature 0.0 \
#     --bon_size 8 \
#     --confidence_threshold 35 \
#     --task_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/models/metamath_llama31_8b  \
#     --reward_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/S2PRM/OpenRLHF/examples/scripts/checkpoint/s2prm_datamistral_modelmistral_bz256_lr1e6_epo1_no_con \
#     --input_data_path GSM8k_test_idx.jsonl \
#     --output_data_path bon_infer_s2prm_datamistral_modelmistral_bz256_lr1e6_epo1_no_con_con35_gsm8k.jsonl \
#     --task_port 8080 \
#     --reward_port 8081 &

# python bon_infer_tempelete.py \
#     --temperature 0.0 \
#     --bon_size 8 \
#     --confidence_threshold 35 \
#     --task_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/models/metamath_llama31_8b  \
#     --reward_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/S2PRM/OpenRLHF/examples/scripts/checkpoint/s2prm_datamistral_modelmistral_bz256_lr1e6_epo1_no_con \
#     --input_data_path GSM8k_test_idx.jsonl \
#     --output_data_path bon_infer_s2prm_datamistral_modelmistral_bz256_lr1e6_epo1_no_con_con35_gsm8k_tempelete.jsonl \
#     --task_port 8080 \
#     --reward_port 8081 &

# wait
# python bon_infer.py \
#     --temperature 0.0 \
#     --bon_size 8 \
#     --confidence_threshold 25 \
#     --task_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/models/metamath_llama31_8b  \
#     --reward_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/S2PRM/OpenRLHF/examples/scripts/checkpoint/s2prm_datamistral_modelmistral_bz256_lr1e6_epo1_no_con \
#     --input_data_path GSM8k_test_idx.jsonl \
#     --output_data_path bon_infer_s2prm_datamistral_modelmistral_bz256_lr1e6_epo1_no_con_con25_gsm8k.jsonl \
#     --task_port 8080 \
#     --reward_port 8081 &

# python bon_infer_tempelete.py \
#     --temperature 0.0 \
#     --bon_size 8 \
#     --confidence_threshold 25 \
#     --task_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/models/metamath_llama31_8b  \
#     --reward_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/S2PRM/OpenRLHF/examples/scripts/checkpoint/s2prm_datamistral_modelmistral_bz256_lr1e6_epo1_no_con \
#     --input_data_path GSM8k_test_idx.jsonl \
#     --output_data_path bon_infer_s2prm_datamistral_modelmistral_bz256_lr1e6_epo1_no_con_con25_gsm8k_tempelete.jsonl \
#     --task_port 8080 \
#     --reward_port 8081 &

# wait

# python bon_infer.py \
#     --temperature 0.0 \
#     --bon_size 8 \
#     --confidence_threshold 35 \
#     --task_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/models/metamath_llama31_8b  \
#     --reward_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/S2PRM/OpenRLHF/examples/scripts/checkpoint/s2prm_datamistral_modelmistral_bz256_lr1e6_epo1_no_con \
#     --input_data_path GSM8k_test_idx.jsonl \
#     --output_data_path bon_infer_s2prm_datamistral_modelmistral_bz256_lr1e6_epo1_no_con_con35_gsm8k.jsonl \
#     --task_port 8080 \
#     --reward_port 8081 &

# python bon_infer_tempelete.py \
#     --temperature 0.0 \
#     --bon_size 8 \
#     --confidence_threshold 35 \
#     --task_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/models/metamath_llama31_8b  \
#     --reward_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/S2PRM/OpenRLHF/examples/scripts/checkpoint/s2prm_datamistral_modelmistral_bz256_lr1e6_epo1_no_con \
#     --input_data_path GSM8k_test_idx.jsonl \
#     --output_data_path bon_infer_s2prm_datamistral_modelmistral_bz256_lr1e6_epo1_no_con_con35_gsm8k_tempelete.jsonl \
#     --task_port 8080 \
#     --reward_port 8081 &

# wait

# python bon_infer.py \
#     --temperature 0.0 \
#     --bon_size 8 \
#     --confidence_threshold 65 \
#     --task_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/models/metamath_llama31_8b  \
#     --reward_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/S2PRM/OpenRLHF/examples/scripts/checkpoint/s2prm_datamistral_modelmistral_bz256_lr1e6_epo1_no_con \
#     --input_data_path GSM8k_test_idx.jsonl \
#     --output_data_path bon_infer_s2prm_datamistral_modelmistral_bz256_lr1e6_epo1_no_con_con65_gsm8k.jsonl \
#     --task_port 8080 \
#     --reward_port 8081 &

# python bon_infer_tempelete.py \
#     --temperature 0.0 \
#     --bon_size 8 \
#     --confidence_threshold 65 \
#     --task_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/models/metamath_llama31_8b  \
#     --reward_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/S2PRM/OpenRLHF/examples/scripts/checkpoint/s2prm_datamistral_modelmistral_bz256_lr1e6_epo1_no_con \
#     --input_data_path GSM8k_test_idx.jsonl \
#     --output_data_path bon_infer_s2prm_datamistral_modelmistral_bz256_lr1e6_epo1_no_con_con65_gsm8k_tempelete.jsonl \
#     --task_port 8080 \
#     --reward_port 8081 &

# wait

# python bon_infer.py \
#     --temperature 0.0 \
#     --bon_size 8 \
#     --confidence_threshold 25 \
#     --task_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/models/metamath_llama31_8b  \
#     --reward_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/S2PRM/OpenRLHF/examples/scripts/checkpoint/s2prm_datamistral_modelmistral_bz256_lr1e6_epo1_no_con \
#     --input_data_path GSM8k_test_idx.jsonl \
#     --output_data_path bon_infer_s2prm_datamistral_modelmistral_bz256_lr1e6_epo1_no_con_con25_gsm8k.jsonl \
#     --task_port 8080 \
#     --reward_port 8081 &

# python bon_infer_tempelete.py \
#     --temperature 0.0 \
#     --bon_size 8 \
#     --confidence_threshold 25 \
#     --task_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/models/metamath_llama31_8b  \
#     --reward_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/S2PRM/OpenRLHF/examples/scripts/checkpoint/s2prm_datamistral_modelmistral_bz256_lr1e6_epo1_no_con \
#     --input_data_path GSM8k_test_idx.jsonl \
#     --output_data_path bon_infer_s2prm_datamistral_modelmistral_bz256_lr1e6_epo1_no_con_con25_gsm8k_tempelete.jsonl \
#     --task_port 8080 \
#     --reward_port 8081 &

# wait
# python bon_infer.py \
#     --temperature 0.0 \
#     --bon_size 8 \
#     --confidence_threshold 50 \
#     --task_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/models/metamath_llama31_8b  \
#     --reward_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/S2PRM/OpenRLHF/examples/scripts/checkpoint/s2prm_datamistral_modelmistral_bz256_lr2e6_epo1_no_con \
#     --input_data_path GSM8k_test_idx.jsonl \
#     --output_data_path bon_infer_s2prm_datamistral_modelmistral_bz256_lr2e6_epo1_no_con_con50_gsm8k.jsonl \
#     --task_port 8080 \
#     --reward_port 8081 &

# wait

# python bon_infer.py \
#     --temperature 0.0 \
#     --bon_size 8 \
#     --confidence_threshold 48 \
#     --task_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/models/metamath_llama31_8b  \
#     --reward_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/S2PRM/OpenRLHF/examples/scripts/checkpoint/s2prm_datamistral_modelmistral_bz256_lr2e6_epo1_no_con \
#     --input_data_path GSM8k_test_idx.jsonl \
#     --output_data_path bon_infer_s2prm_datamistral_modelmistral_bz256_lr2e6_epo1_no_con_con48_gsm8k.jsonl \
#     --task_port 8080 \
#     --reward_port 8081 &

# python bon_infer.py \
#     --temperature 0.0 \
#     --bon_size 8 \
#     --confidence_threshold 45 \
#     --task_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/models/metamath_llama31_8b  \
#     --reward_model_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/S2PRM/OpenRLHF/examples/scripts/checkpoint/s2prm_datamistral_modelmistral_bz256_lr2e6_epo1_no_con \
#     --input_data_path GSM8k_test_idx.jsonl \
#     --output_data_path bon_infer_s2prm_datamistral_modelmistral_bz256_lr2e6_epo1_no_con_con45_gsm8k.jsonl \
#     --task_port 8080 \
#     --reward_port 8081 &

# wait