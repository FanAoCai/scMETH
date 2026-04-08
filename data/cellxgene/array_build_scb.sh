#!/bin/bash
QUERY_PATH="/mnt/afan/scGEPOP/data/cellxgene/query_list.txt"

# 读取 QUERY_PATH 中的每一行作为 query_name，并循环处理
while IFS= read -r query_name; do
    # 跳过空行
    if [[ -z "$query_name" ]]; then
        continue
    fi

    DATA_PATH="/mnt/afan/scGEPOP/data/cellxgene/dataset/${query_name}"

    if [[ ! -d "$DATA_PATH" ]]; then
        echo "Warning: ${DATA_PATH} not exists, Skipping ${query_name}."
        continue
    fi

    OUTPUT_PATH="/mnt/afan/scGEPOP/data/cellxgene/scb_file/${query_name}"
    VOCAB_PATH="/mnt/afan/scGEPOP/scgepop/tokenizer/default_census_vocab.json"

    echo "processing ${query_name}"
    N=50000

    mkdir -p "$OUTPUT_PATH"

    echo "downloading to ${OUTPUT_PATH}"

    python build_large_scale_data.py \
        --input_dir "$DATA_PATH" \
        --output_dir "$OUTPUT_PATH" \
        --vocab_file "$VOCAB_PATH" \
        --N "$N"

done < "$QUERY_PATH"