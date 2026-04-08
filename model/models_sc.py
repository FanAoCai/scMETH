# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed
from model.moe_block import MoEBlock


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, vocab, moe, cls_eth, eth_cls_num, criterion, hidden_dim=256,
                 pad_token='<pad>', embed_dim=512, depth=24, num_heads=8, 
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4, num_experts=8, norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # scMAE encoder specifics
        self.genes_encoder = GeneValueEncoder(len(vocab), embed_dim, vocab[pad_token])
        self.expr_encoder = ExprValueEncoder(embed_dim)

        if moe:
            self.blocks = nn.ModuleList([
            MoEBlock(embed_dim, num_heads, num_experts, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer) 
            if (i % 2 == 1) else 
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        else:
            self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # scEthnicity_cls decoder specifics
        self.eth_docoder = nn.Sequential(
            nn.Linear(embed_dim*2, hidden_dim),
            norm_layer(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, eth_cls_num)
            )
        
        self.criterion = criterion
        self.cls_eth = cls_eth
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # scMAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.genes_decoder = GeneValueEncoder(len(vocab), decoder_embed_dim, vocab[pad_token])

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)

        self.decoder_expr = nn.Sequential(
            nn.Linear(decoder_embed_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1)
            )
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss
        self.moe = moe

        # self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed genes and expressions
        genes = x['genes']
        expr = x['expr']

        genes_embs = self.genes_encoder(genes)
        expr_embs = self.expr_encoder(expr)

        x = genes_embs + expr_embs

        cls_tokens = x[:, :1, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x[:, 1:, :], mask_ratio)

        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        loss_moe = 0
        if self.moe:
            for idx, blk in enumerate(self.blocks):
                if (idx % 2 == 1):
                    output = blk(x)
                    x = output[0]
                    loss_moe += output[1]
                else:
                    x = blk(x)
        else:
            for blk in self.blocks:
                x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore, genes, loss_moe

    def forward_decoder(self, x, ids_restore, genes):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.genes_decoder(genes)

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x) 

        # remove cls token
        x = x[:, 1:, :]

        x = self.decoder_expr(x).squeeze(-1)

        return x
    
    def forward_decoder_eth(self, x):
        cls_token = x[:, 0, :]
        cell_feature = x[:, 1:, :].mean(dim=1)
        
        x = torch.cat([cls_token, cell_feature], dim=-1)

        x = self.eth_docoder(x)

        return x

    def forward_loss(self, samples, pred, mask):
        target = samples['expr'][:, 1:]
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss
    
    def forward_eth(self, samples, pred_eth):
        target = samples['ethnicity']

        loss = self.criterion(pred_eth, target)

        _, result = torch.max(pred_eth, dim=1)
        correct = (result == target)
        acc = correct.float().mean()

        return loss, acc

    # add return 'latent'
    def forward(self, samples, mask_ratio=0.75):
        latent, mask, ids_restore, genes, loss_moe = self.forward_encoder(samples, mask_ratio)

        # MAE Task
        pred = self.forward_decoder(latent, ids_restore, genes)
        loss_mask = self.forward_loss(samples, pred, mask)
        
        # Ethnic Classification Task
        loss_eth = 0
        acc = 0
        if self.cls_eth:
            pred_eth = self.forward_decoder_eth(latent)
            loss_eth, acc = self.forward_eth(samples, pred_eth)

        return latent, loss_mask, loss_moe, pred, mask, loss_eth, acc


class GeneValueEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.enc_norm(x)
        return x
    

class ExprValueEncoder(nn.Module):
    """
    Encode real number values to a vector using neural nets projection.
    """

    def __init__(self, d_model: int, dropout: float = 0.5, max_value: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(1, d_model)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.max_value = max_value

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len]
        """
        # TODO: test using actual embedding layer if input is categorical
        # expand last dimension
        x = x.unsqueeze(-1)
        # clip x to [-inf, max_value]
        x = torch.clamp(x, max=self.max_value)
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        x = self.norm(x)
        return self.dropout(x)


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        hidden_dim=256, pad_token='<pad>', embed_dim=512, depth=8, num_heads=8, 
        decoder_embed_dim=256, decoder_depth=4, decoder_num_heads=8,
        mlp_ratio=4, num_experts=8, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        hidden_dim=256, pad_token='<pad>', embed_dim=512, depth=12, num_heads=16,
        decoder_embed_dim=256, decoder_depth=4, decoder_num_heads=8,
        mlp_ratio=4, num_experts=8, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        hidden_dim=256, pad_token='<pad>', embed_dim=768, depth=16, num_heads=16, 
        decoder_embed_dim=256, decoder_depth=4, decoder_num_heads=8,
        mlp_ratio=4, num_experts=8, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
