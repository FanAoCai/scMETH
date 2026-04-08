QUERY_PATH="/mnt/afan/scGEPOP/data/cellxgene/query_list.txt"
VOCAB_PATH="/mnt/afan/scGEPOP/scgepop/tokenizer/default_census_vocab.json"

while IFS= read -r query_name; do
    if [[ -z "$query_name" ]]; then
        continue
    fi

    echo "***processing ${query_name}***"
    DATASET="/mnt/afan/scGEPOP/data/cellxgene/scb_file/${query_name}/all_counts"

    if [[ ! -d "$DATASET" ]]; then
        echo "Warning: ${DATASET} not exists, Skipping ${query_name}."
        continue
    fi

    python -c "import torch; print(torch.version.cuda)"
    python process_allcounts.py \
        --data_source "$DATASET" \
        --vocab_path "${VOCAB_PATH}" |
        awk '{ print strftime("[%Y-%m-%d %H:%M:%S]"), $0; fflush(); }'
done < "$QUERY_PATH"

