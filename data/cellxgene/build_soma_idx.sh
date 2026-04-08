#!/bin/sh
# output directory for the index 
OUTPUT_DIR='/mnt/afan/scGEPOP/data/cellxgene/cell_index'
QUERY_LIST='/mnt/afan/scGEPOP/data/cellxgene/query_list.txt'

# ensure output dir exists
mkdir -p "$OUTPUT_DIR"

# run from script directory so relative paths to build_soma_idx.py are correct
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# Read lines robustly (preserve spaces), skip empty lines and comments starting with '#'
while IFS= read -r QUERY || [ -n "$QUERY" ]; do
    case "$QUERY" in
        ""|\#*) continue ;; # skip empty or comment lines
    esac

    echo "building index for: $QUERY"
    # pass named flags and quote variables to preserve spaces
    python3 ./build_soma_idx.py --query_name "$QUERY" --output_dir "$OUTPUT_DIR"
done < "$QUERY_LIST"