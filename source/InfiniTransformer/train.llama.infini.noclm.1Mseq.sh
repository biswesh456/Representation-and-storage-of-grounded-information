#export CUDA_VISIBLE_DEVICES=0

# DEBUG=true 
accelerate launch --num_processes=1 --mixed_precision='bf16' \
    train.llama.infini.noclm.py \
    --model_name_or_path='meta-llama/Meta-Llama-3.1-8B-Instruct' \
    --segment_length=2048 \
    --block_size=8192 \
    --dataset_name='wikitext' \
    --dataset_config_name='wikitext-2-raw-v1' \
    --train_file='~/data/temporal/' \
    --per_device_train_batch_size=1 \
    --per_device_eval_batch_size=1 \
    --output_dir='../models/llama-3.1-8b-infini-noclm-gated' \
    --checkpointing_steps=1000 \
    --num_train_epochs=1 \
    --learning_rate=1e-4 \
    --seed=42 \
    --low_cpu_mem_usage \
    --report_to='wandb' \
    --preprocessing_num_workers=64 \
    --with_tracking \
