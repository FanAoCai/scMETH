INDEX_PATH="/mnt/afan/scGEPOP/data/cellxgene/cell_index"
QUERY_PATH="/mnt/afan/scGEPOP/data/cellxgene/query_list.txt"
DATA_PATH="/mnt/afan/scGEPOP/data/cellxgene/dataset"

cd "$DATA_PATH"

# 逐行读取 query_list.txt
while IFS= read -r query_name; do
    # 跳过空行
    [[ -z "$query_name" ]] && continue
    echo "downloading ${query_name}"
    /mnt/afan/scGEPOP/data/cellxgene/download_partition.sh "${query_name}" "${INDEX_PATH}" "${DATA_PATH}"
done < "$QUERY_PATH"