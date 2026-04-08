#!/usr/bin/env bash
set -euo pipefail

# Usage: download_partition.sh <query_name> <index_dir> <output_dir> [max_partition_size]
QUERY="${1:-African American}"
INDEX_DIR="${2:-/mnt/afan/scGEPOP/data/cellxgene/cell_index}"
OUTPUT_DIR="${3:-/mnt/afan/scGEPOP/data/cellxgene/dataset}"
MAX_PARTITION_SIZE="${4:-50000}"

if [ -z "$QUERY" ] || [ -z "$INDEX_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: $0 <query_name> <index_dir> <output_dir> [max_partition_size]" >&2
    exit 2
fi

mkdir -p "$OUTPUT_DIR"

INDEX_FILE="$INDEX_DIR/$QUERY.idx"
if [ ! -f "$INDEX_FILE" ]; then
    echo "Index file not found: $INDEX_FILE" >&2
    exit 1
fi

# count lines (records)
total_num=$(wc -l < "$INDEX_FILE" | awk '{print $1}')
if [ -z "$total_num" ] || [ "$total_num" -eq 0 ]; then
    echo "No records in $INDEX_FILE, nothing to download.";
    exit 0
fi

# compute number of partitions (ceiling division)
total_partitions=$(( (total_num + MAX_PARTITION_SIZE - 1) / MAX_PARTITION_SIZE ))
last_idx=$(( total_partitions - 1 ))

echo "Query: $QUERY"
echo "Index: $INDEX_FILE"
echo "Total records: $total_num"
echo "Partitions: $total_partitions (indexes 0..$last_idx, max size: $MAX_PARTITION_SIZE)"

for i in $(seq 0 "$last_idx"); do
    echo "downloading partition ${i}/${last_idx} for '${QUERY}'"

    # call python script with quoted args
    python3 /mnt/afan/scGEPOP/data/cellxgene/download_partition.py \
        --query_name "$QUERY" \
        --index_dir "$INDEX_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --partition_idx "$i" \
        --max_partition_size "$MAX_PARTITION_SIZE"
done