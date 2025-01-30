CUDA_VISIBLE_DEVICES=$1 python -m vllm.entrypoints.openai.api_server \
    --model $2 \
    --trust-remote-code \
    --port $3 \
    --tensor-parallel-size 1 \
    --enforce-eager \
    --max-model-len 4096 \
    --tokenizer $4 \
    --max-logprobs 20 \
    --device "cuda"