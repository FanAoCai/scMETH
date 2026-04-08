import argparse
from pathlib import Path
from scgepop.tokenizer import GeneVocab, random_mask_value
import sys
from datasets import Dataset, load_dataset
import os

sys.path.insert(0, "../")

parser = argparse.ArgumentParser()
parser.add_argument(
    "-d",
    "--data_source",
    type=str,
    # required=True,
    default="/mnt/afan/scGEPOP/data/cellxgene/scb_file/African American/all_counts",
    help='The name of the data source (currently support "scvi" datasets), or the '
    "path to the data file.",
)

# settings for tokenizer
parser.add_argument(
    "--pad_token",
    type=str,
    default="<pad>",
    help="The token to use for padding. Default is <pad>.",
)
parser.add_argument(
    "--input_style",
    type=str,
    choices=["normed_raw", "log1p", "binned"],
    default="binned",
    help="The style of the input data. Default is binned.",
)
parser.add_argument(
    "--input_emb_style",
    type=str,
    choices=["category", "continuous", "scaling"],
    default="continuous",
    help="The style of the input embedding. Default is continuous.",
)
parser.add_argument(
    "--n_bins",
    type=int,
    default=51,
    help="The number of bins to use for the binned input style. Default is 51.",
)
# omit the args for MLM and MVC, will always use them by default
parser.add_argument(
    "--vocab_path",
    type=str,
    default="/mnt/afan/scGEPOP/scgepop/tokenizer/default_census_vocab.json",
    help="Path to the vocabulary file.",
)

args = parser.parse_args()

if args.input_style == "binned":
    if args.input_emb_style == "scaling":
        raise ValueError("input_emb_style `scaling` is not supported for binned input.")
elif args.input_style == "log1p" or args.input_style == "normed_raw":
    if args.input_emb_style == "category":
        raise ValueError(
            "input_emb_style `category` is not supported for log1p or normed_raw input."
        )

if args.input_emb_style == "category":
    args.mask_value = args.n_bins + 1
    args.pad_value = args.n_bins  # for padding gene expr values
    n_input_bins = args.n_bins + 2
else:
    args.mask_value = -1
    args.pad_value = -2
    n_input_bins = args.n_bins


def _map_append_cls(dataset: Dataset) -> Dataset:
    dataset = dataset.map(
        lambda example: {
            "genes": [vocab["<cls>"]] + example["genes"],
            "expressions": [args.pad_value] + example["expressions"],
        },
        # batched=True,  # not using since then the map func needs to loop
        num_proc=len(os.sched_getaffinity(0)),
    )

    return dataset


special_tokens = [args.pad_token, "<cls>", "<eoc>"]

parquet_files = [str(f) for f in Path(args.data_source).glob("*.parquet")]
cache_dir = Path(args.data_source).parent / "cache"
vocab = GeneVocab.from_file(Path(args.vocab_path))
for s in special_tokens:
    if s not in vocab:
        vocab.append_token(s)


# load or make the dataset w/ <cls> appended at the beginning
cls_prefix_datatable = Path(args.data_source) / "cls_prefix_data.parquet"
if not cls_prefix_datatable.exists():
    print("preparing cls prefix dataset")
    raw_dataset = load_dataset(
        "parquet",
        data_files=parquet_files,
        split="train",
        cache_dir=str(cache_dir),
    )
    raw_dataset = _map_append_cls(raw_dataset)
    raw_dataset.to_parquet(str(cls_prefix_datatable))
raw_dataset = load_dataset(
    "parquet",
    data_files=str(cls_prefix_datatable),
    split="train",
    cache_dir=str(cache_dir),
)
