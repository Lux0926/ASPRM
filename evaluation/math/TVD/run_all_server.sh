
# base model
bash run_task_server.sh 0 /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/models/metamath_llama31_8b 8080 /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/models/metamath_llama31_8b &

# PRM model
bash run_reward_server_mistral.sh 1 /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/S2PRM/OpenRLHF/examples/scripts/checkpoint/s2prm_datamistralv3_modelmistral_bz256_lr1e6_epo1_no_con 8081 &


# base model
bash run_task_server.sh 2 /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/models/metamath_llama31_8b 8090 /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/models/metamath_llama31_8b &

# PRM model
bash run_reward_server_llama.sh 3 /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/S2PRM/OpenRLHF/examples/scripts/checkpoint/s2prm_datamistralv3_modelllama_bz256_lr2e6_epo1_no_con 8083 &

