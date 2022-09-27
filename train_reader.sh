#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --job-name=FID_train
#SBATCH --nodelist=ink-gary
#SBATCH --output ./jobs/%j.out
#SBATCH --gres=gpu:1
#SBATCH -n 1

train_data_path="./data/cnn_dailymail_train_hypo.jsonl"
dev_data_path="./data/cnn_dailymail_val_hypo.jsonl"
test_data_path="./data/cnn_dailymail_test_hypo.jsonl"
NGPU=1

nvidia-smi

model_type='bart'
model_size="base"
name="basic"
if [ ${model_type} == 't5' ]; then
        text_maxlength=512
elif [ ${model_type} == 'bart' ]; then
        text_maxlength=1024
else
        echo "model type not supported"
        exit 1
fi


echo "model type: ${model_type}"
echo "model size: ${model_size}"
echo "name: ${name}"
echo "text_maxlength: ${text_maxlength}"

# torchrun \
        # --nproc_per_node=$NGPU \
        python train_reader.py \
        --name "${model_type}-${model_size}-${name}" \
        --train_data ${train_data_path} \
        --eval_data ${dev_data_path} \
        --model_type ${model_type} \
        --model_size ${model_size} \
        --text_maxlength ${text_maxlength} \
        --checkpoint_dir checkpoint \
        --lr 0.00005 \
        --optim adamw \
        --scheduler linear \
        --weight_decay 0.01 \
        --per_gpu_batch_size 1 \
        --n_context 10 \
        --total_step 15000 \
        --warmup_step 1000 \
        --main_port 19002 \
