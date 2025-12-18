set -e
set -x

if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

OUTPUT_DIR=outputs/qwen3_8b_ft/recom_reply_format_sft/2025-12-19
mkdir -p $OUTPUT_DIR

TRAIN_CONFIG_FILE=qwen3_8b_ft/recom_reply_format_sft/train.yaml
cp $TRAIN_CONFIG_FILE $OUTPUT_DIR/train.yaml

ACCELERATE_CONFIG_FILE=qwen3_8b_ft/v100_accelerate_config.yml

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 axolotl train $TRAIN_CONFIG_FILE \
    --launcher accelerate \
    -- --config_file=$ACCELERATE_CONFIG_FILE \
    2>&1 | tee $OUTPUT_DIR/train.log
