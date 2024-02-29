#!/bin/bash

export OMP_NUM_THREADS=20
llama_path=$1
output_dir=$2
data_dir=$3
ds_config=$4

deepspeed \
    --master_port=9943 \
    rear/train.py \
    --deepspeed $4 \
    --model_name_or_path $llama_path \
    --data ${data_dir}/warm_up_data.json  \
    --is_warm_up true \
    --rank_beta 1 \
    --rank_bias 0.7 \
    --proj_scaler 2.0 \
    --lr_scheduler_type cosine \
    --output_dir $output_dir/warm-up \
    --overwrite_output_dir \
    --warmup_ratio 0.03 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 2 \
    --logging_steps 10 \
    --learning_rate 1e-6 \
    --num_train_epochs 2 \
    --bf16

deepspeed \
    --master_port=9943 \
    rear/train.py \
    --deepspeed $4 \
    --model_name_or_path $output_dir/warm-up \
    --data ${data_dir}/training_data.json  \
    --is_warm_up false \
    --rank_beta 1 \
    --rank_bias 0.8 \
    --bce_bias 0.3 \
    --proj_scaler 2.0 \
    --minor_diff 0.1 \
    --lr_scheduler_type cosine \
    --output_dir $output_dir/rear-llama-7b-hf \
    --overwrite_output_dir \
    --warmup_ratio 0.03 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --logging_steps 10 \
    --learning_rate 1e-6 \
    --num_train_epochs 2 \
    --bf16