#!/bin/bash

# ******single GPU******
output_dir="./output_dir"
CUDA_VISIBLE_DEVICES=1 python main_pretrain.py \
--batch_size 64 \
--epochs 20 \
--model 'mae_vit_base_patch16' \
--mask_ratio 0.4 \
--norm_pix_loss \
--blr 4e-4 \
--data_path '/mnt/afan/scGEPOP/data/cellxgene/scb_file' \
--ethnicity_path '/mnt/afan/scGEPOP/data/cellxgene/query_list.txt' \
--vocab_path '/mnt/afan/scGEPOP/scgepop/tokenizer/default_census_vocab.json' \
--output_dir=$output_dir \
--valid_ratio 0.03 \
--max_seq_len 1536 \
--dist_eval \


# ******multiple GPUs******
# output_dir="./output_dir"
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=4 --master_port='29502' main_pretrain.py \
# --batch_size 128 \
# --epochs 300 \
# --model 'mae_vit_large_patch16' \
# --mask_ratio 0.4 \
# --norm_pix_loss \
# --moe \
# --blr 1e-3 \
# --data_path '/mnt/afan/scGEPOP/data/cellxgene/scb_file' \
# --ethnicity_path '/mnt/afan/scGEPOP/data/cellxgene/query_list.txt' \
# --vocab_path '/mnt/afan/scGEPOP/scgepop/tokenizer/default_census_vocab.json' \
# --output_dir=$output_dir \
# --load_current_pretrained_weight '/mnt/afan/Path_SSL/pretrained_weight/mae_pretrain_vit_large.pth' \
# --valid_ratio 0.03 \
# --max_seq_len 1536 \
# --dist_eval \