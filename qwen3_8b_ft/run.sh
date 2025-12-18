set -e
set -x

# docker run -it --ipc=host --shm-size=32g --privileged --gpus all \
#     -e TOKENIZERS_PARALLELISM=false \
#     -e HF_ENDPOINT=https://hf-mirror.com \
#     -v /mnt/cephfs2/:/mnt/cephfs2 \
#     -v /home/peichao.murphy/.cache/huggingface:/root/.cache/huggingface \
#     -w /workspace \
#     harbor.bigo.sg/bigo_ai/nlp/axolotl-v100 \
#     bash

if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

# 启动训练
CUDA_VISIBLE_DEVICES=0,1,2,3 axolotl \
    train examples/qwen3/8b-lora-ds-chat-v100.yaml \
    --launcher accelerate -- --config_file=accelerate_configs/v100_config.yml \
    2>&1 | tee train.log

# debug 参数
#  --debug-text-only=False 
#  --debug-num-examples=0 
#  --shard=False

# merge lora
axolotl merge-lora examples/qwen3/8b-lora-chat-v100.yaml \
 --lora-model-dir ./outputs/qwen3-8b-lora-chat/bigolive_human_data_2025-12-05/checkpoint-3213


CUDA_VISIBLE_DEVICES=0，4,5,6,7 axolotl \
    train examples/qwen3/8b-lora-dpo-v100.yaml \
    --launcher accelerate -- --config_file=accelerate_configs/v100_config.yml \
    2>&1 | tee dpo_train.log
