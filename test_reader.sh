#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --job-name=bash
#SBATCH --nodelist=ink-gary
#SBATCH --output ./jobs/%j.out
#SBATCH --gres=gpu:1
#SBATCH -n 1

train_data_path="./data/cnn_dailymail__hypo.jsonl"
dev_data_path="./data/cnn_dailymail_val_hypo.jsonl"
test_data_path="./data/cnn_dailymail_test_hypo.jsonl"
nvidia-smi

model_type='bart'
if [ ${model_type} == 't5' ]; then
        text_maxlength=512
elif [ ${model_type} == 'bart' ]; then
        text_maxlength=1024
else
        echo "model type not supported"
        exit 1
fi

python test_reader.py \
        --model_type ${model_type} \
        --name titan_try \
        --model_path checkpoint/titan_try/checkpoint/step-5000 \
        --eval_data ${test_data_path} \
        --per_gpu_batch_size 2 \
        --n_context 10 \
        --text_maxlength ${text_maxlength} \
        --checkpoint_dir checkpoint \
        --write_results \
        --main_port 19000 \
