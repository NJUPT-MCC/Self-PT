source activate YOUR_ENVIRONMENT
export CUDA_VISIBLE_DEVICES="0"

# average across 5 seeds
for seed in 13 21 42 87 100
do
    python YOUR_PROJECT/.../vqa.py \
    --subsample --num_data 16  --dataseed $seed  --output '.../your_log_file/' --load '.../Epoch30_base' \
    --tasks "vqa"  \
    --use_prompt --pre_seq_len 5 --prompt_type 'hyper_phm_new' \
    --use_single_adapter --use_compacter
done
