CUDA_VISIBLE_DEVICES=$1 python -m vllm.entrypoints.openai.api_server \
    --model $2 \
    --trust-remote-code \
    --served-model-name soft-prm \
    --port $3 \
    --max-model-len 8192 \
    --tensor-parallel-size 1 \
    --enforce-eager \
    --task embedding \
    --pooling-type STEP \
    --pooling-step-tag-id 116624 \
    --pooling-returned-token-ids 7820 7740 