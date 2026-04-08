from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import torch
import numpy as np


@dataclass
class DataCollator:
    """
    Data collator for the mask value learning task. It pads the sequences to
    the maximum length in the batch and masks the gene expression values.

    Args:
        do_padding (:obj:`bool`): whether to pad the sequences to the max length.
        pad_token_id (:obj:`int`, optional): the token id to use for padding.
            This is required if do_padding is True.
        pad_value (:obj:`int`): the value to use for padding the expression
            values to the max length.
        do_mlm (:obj:`bool`): whether to do masking with MLM.
        do_binning (:obj:`bool`): whether to bin the expression values.
        mlm_probability (:obj:`float`): the probability of masking with MLM.
        mask_value (:obj:`int`): the value to fill at the expression postions
            that are masked.
        max_length (:obj:`int`, optional): the maximum length of the sequences.
            This is required if do_padding is True.
        sampling (:obj:`bool`): whether to do sampling instead of truncation if
            length > max_length.
        reserve_keys (:obj:`List[str]`, optional): a list of keys in the examples
            to reserve in the output dictionary. Default to []. These fields
            will be kept unchanged in the output.
        keep_first_n_tokens (:obj:`int`): the number of tokens in the beginning
            of the sequence to keep unchanged from sampling. This is useful when
            special tokens have been added to the beginning of the sequence.
            Default to 1.
        data_style (:obj:`str`): the style of the data. If "pcpt", the data is
            masked and padded for perception training. If "gen", only the gene
            tokens are provided, but not the expression values, for pure generative
            training setting. If "both", the output will contain both fields above.
            Choices: "pcpt", "gen", "both". Default to "pcpt".
    """

    pad_token: Optional[int] = None
    pad_value: int = 0
    max_length: Optional[int] = None
    sampling: bool = True
    keep_first_n_tokens: int = 1


    def __call__(
        self, examples: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:

        if not isinstance(examples[0], Mapping):
            return NotImplementedError

        device = examples[0]["genes"].device

        max_ori_len = max(len(example["genes"]) for example in examples)
        _max_length = self.max_length if max_ori_len >= self.max_length else max_ori_len

        # pad and truncate
        padded_pcpt_genes = []
        padded_pcpt_expressions = []
        padded_ethnicitys = []
        for i in range(len(examples)):
            genes = examples[i]["genes"]
            expressions = examples[i]["expressions"]
            ethnicity = examples[i]["ethnicity"]
            padded_ethnicitys.append(ethnicity)
            expressions[self.keep_first_n_tokens :] = binning(
                row=expressions[self.keep_first_n_tokens :],
                n_bins=51,  # FIXME: replace with self.n_bins
            )
            genes, expressions = self._sample_or_truncate_plus_pad(
                genes, expressions, _max_length
            )
            padded_pcpt_genes.append(genes)
            padded_pcpt_expressions.append(expressions)

        padded_pcpt_genes = torch.stack(padded_pcpt_genes, dim=0).to(device)
        padded_pcpt_expressions = torch.stack(padded_pcpt_expressions, dim=0).to(device)
        padded_ethnicitys = torch.stack(padded_ethnicitys, dim=0).to(device)

        data_dict = {
            "genes": padded_pcpt_genes,
            "expr": padded_pcpt_expressions,
            "ethnicity": padded_ethnicitys
        }

        return data_dict


    def _sample_or_truncate_plus_pad(
        self,
        genes: torch.LongTensor,
        expressions: torch.Tensor,
        max_length: int,
    ) -> Tuple[torch.LongTensor, torch.Tensor]:
        assert len(genes) == len(expressions)
        if len(genes) == max_length:
            return genes, expressions
        if len(genes) > max_length:  # sample or truncate
            if self.sampling:
                return self._sample(genes, expressions, max_length)
            else:
                return genes[:max_length], expressions[:max_length]
        else:  # pad
            return self._pad(genes, expressions, max_length)

    def _sample(
        self,
        genes: torch.LongTensor,
        expressions: torch.Tensor,
        max_length: int,
    ) -> Tuple[torch.LongTensor, torch.Tensor]:
        # NOTE: the fastest way to sample in torch has been benchmarked here
        # https://discuss.pytorch.org/t/torch-equivalent-of-numpy-random-choice/16146/19
        # it shows the randperm on gpu is the fastest.
        # NOTE: also, the current implementation permute the orders of the genes
        # and expressions, although it is probably a nice argmentation.
        device = genes.device
        if self.keep_first_n_tokens == 0:
            indices = torch.randperm(len(genes), device=device)[:max_length]
            return genes[indices], expressions[indices]

        # keep the first n tokens unchanged
        _n = self.keep_first_n_tokens
        indices = torch.randperm(len(genes) - _n, device=device)[: max_length - _n]
        indices = torch.cat([torch.arange(_n), indices + _n], dim=0)
        return genes[indices], expressions[indices]

    def _pad(
        self,
        genes: torch.LongTensor,
        expressions: torch.Tensor,
        max_length: int,
    ):
        device = genes.device
        genes = torch.cat(
            [
                genes,
                torch.full(
                    (max_length - len(genes),),
                    self.pad_token,
                    dtype=genes.dtype,
                    device=device,
                ),
            ]
        )
        expressions = torch.cat(
            [
                expressions,
                torch.full(
                    (max_length - len(expressions),),
                    self.pad_value,
                    dtype=expressions.dtype,
                    device=device,
                ),
            ]
        )
        return genes, expressions


def _digitize(x: np.ndarray, bins: np.ndarray, side="one") -> np.ndarray:
    """
    Digitize the data into bins. This method spreads data uniformly when bins
    have same values.

    Args:

    x (:class:`np.ndarray`):
        The data to digitize.
    bins (:class:`np.ndarray`):
        The bins to use for digitization, in increasing order.
    side (:class:`str`, optional):
        The side to use for digitization. If "one", the left side is used. If
        "both", the left and right side are used. Default to "one".

    Returns:

    :class:`np.ndarray`:
        The digitized data.
    """
    assert x.ndim == 1 and bins.ndim == 1

    left_digits = np.digitize(x, bins)
    if side == "one":
        return left_digits

    right_difits = np.digitize(x, bins, right=True)

    rands = np.random.rand(len(x))  # uniform random numbers

    digits = rands * (right_difits - left_digits) + left_digits
    digits = np.ceil(digits).astype(np.int64)
    return digits


def binning(
    row: Union[np.ndarray, torch.Tensor], n_bins: int
) -> Union[np.ndarray, torch.Tensor]:
    """Binning the row into n_bins."""
    dtype = row.dtype
    return_np = False if isinstance(row, torch.Tensor) else True
    row = row.cpu().numpy() if isinstance(row, torch.Tensor) else row
    # TODO: use torch.quantile and torch.bucketize

    if row.min() <= 0:
        non_zero_ids = row.nonzero()
        non_zero_row = row[non_zero_ids]
        bins = np.quantile(non_zero_row, np.linspace(0, 1, n_bins - 1))
        non_zero_digits = _digitize(non_zero_row, bins)
        binned_row = np.zeros_like(row, dtype=np.int64)
        binned_row[non_zero_ids] = non_zero_digits
    else:
        bins = np.quantile(row, np.linspace(0, 1, n_bins - 1))
        binned_row = _digitize(row, bins)
    return torch.from_numpy(binned_row) if not return_np else binned_row.astype(dtype)