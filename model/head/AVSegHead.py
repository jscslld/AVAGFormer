from einops import rearrange, repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils import build_transformer, build_positional_encoding, build_fusion_block, build_generator,DownsampleConv,ConvTranspose2d
from model.utils.fusion_block import NewCrossModalMixer
from ops.modules import MSDeformAttn
from torch.nn.init import normal_
from torch.nn.functional import interpolate
# from transformers import CLIPProcessor, CLIPModel
import torchvision.transforms as transforms


from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Optional
import math
from einops import rearrange, repeat
import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------- SimpleFPN -----------
class SimpleFPN(nn.Module):
    def __init__(self, in_dims, out_dim):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_dim, out_dim, 1) for in_dim in in_dims
        ])
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_dim, out_dim, 3, padding=1) for _ in in_dims
        ])

    def forward(self, feats):
        # feats: list of [B, C_i, H_i, W_i], from high to low resolution
        laterals = [l_conv(f) for l_conv, f in zip(self.lateral_convs, feats)]
        for i in range(len(laterals)-1, 0, -1):
            upsample = F.interpolate(laterals[i], size=laterals[i-1].shape[-2:], mode='bilinear', align_corners=False)
            laterals[i-1] += upsample
        outs = [o_conv(l) for o_conv, l in zip(self.output_convs, laterals)]
        return outs  # list of [B, out_dim, H_i, W_i]

# ----------- StructureFiLM -----------
class StructureFiLM(nn.Module):
    def __init__(self, feature_dim, structure_dim):
        super().__init__()
        self.gamma = nn.Linear(structure_dim, feature_dim)
        self.beta = nn.Linear(structure_dim, feature_dim)

    def forward(self, x, structure_info):
        gamma = self.gamma(structure_info).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta(structure_info).unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta

# ----------- SimpleFiLMCrossAttentionDecoder -----------
class SimpleFiLMCrossAttentionDecoder(nn.Module):
    def __init__(self, in_dim, audio_dim, structure_dim=None, out_dim=16, num_heads=4, num_scales=3):
        super().__init__()
        self.num_scales = num_scales
        self.proj_vis = nn.ModuleList([nn.Conv2d(in_dim, out_dim, 1) for _ in range(num_scales)])
        self.proj_audio = nn.ModuleList([nn.Linear(audio_dim, out_dim) for _ in range(num_scales)])
        self.cross_attn = nn.ModuleList([nn.MultiheadAttention(out_dim, num_heads) for _ in range(num_scales)])
        self.conv_fuse = nn.ModuleList([nn.Conv2d(out_dim*2, out_dim, 3, padding=1) for _ in range(num_scales)])
        self.norms = nn.ModuleList([nn.GroupNorm(4, out_dim) for _ in range(num_scales)])
        self.relu = nn.ReLU(inplace=True)
        if structure_dim is not None:
            self.film = StructureFiLM(out_dim, structure_dim)
        else:
            self.film = None

    def forward(self, vis_feats, audio_feats, structure_info=None):
        # vis_feats: list of [B, C, H, W], len=num_scales
        # audio_feats: list of [B, T, C_audio], len=num_scales
        outs = []
        for i, (vis, audio_feat) in enumerate(zip(vis_feats, audio_feats)):
            B, C, H, W = vis.shape
            T = audio_feat.shape[1]
            # 1. 投影
            vis_proj = self.proj_vis[i](vis)  # [B, out_dim, H, W]
            audio_proj = self.proj_audio[i](audio_feat)  # [B, T, out_dim]
            # 2. flatten
            vis_flat = vis_proj.flatten(2).permute(2, 0, 1)     # [HW, B, out_dim]
            audio_flat = audio_proj.permute(1, 0, 2)             # [T, B, out_dim]
            # 3. cross-attn
            attn_out, _ = self.cross_attn[i](vis_flat, audio_flat, audio_flat)  # [HW, B, out_dim]
            attn_out = attn_out.permute(1, 2, 0).reshape(B, -1, H, W)          # [B, out_dim, H, W]
            # 4. 融合
            fuse = torch.cat([vis_proj, attn_out], dim=1)
            fuse = self.conv_fuse[i](fuse)
            fuse = self.norms[i](fuse)
            fuse = self.relu(fuse)
            if self.film is not None and structure_info is not None:
                fuse = self.film(fuse, structure_info)
            outs.append(fuse)
        # 上采样到最高分辨率并融合
        out = outs[0]
        for i in range(1, len(outs)):
            out = out + F.interpolate(outs[i], size=out.shape[-2:], mode='bilinear', align_corners=False)
        return out  # [B, out_dim, H, W]


class QueryConditionedMaskHead(nn.Module):
    def __init__(self, feat_dim, query_dim, num_classes):
        super().__init__()
        self.query_proj = nn.Linear(query_dim, feat_dim)
        self.mask_head = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feat_dim, num_classes, 1)
        )

    def forward(self, feat, query):  # feat: [B, C, H, W], query: [B, Q, C]
        B, C, H, W = feat.shape
        Q = query.shape[1]
        # query: [B, Q, query_dim] -> [B, Q, C]
        query_proj = self.query_proj(query)  # [B, Q, C]
        # 扩展到空间
        query_map = query_proj.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, H, W)  # [B, Q, C, H, W]
        feat_map = feat.unsqueeze(1).expand(-1, Q, -1, -1, -1)                      # [B, Q, C, H, W]
        fusion = feat_map + query_map  # [B, Q, C, H, W]
        fusion = fusion.view(B*Q, C, H, W)
        mask_logits = self.mask_head(fusion)  # [B*Q, num_classes, H, W]
        mask_logits = mask_logits.view(B, Q, -1, H, W)  # -1就是num_classes
        return mask_logits  # [B, Q, num_classes, H, W]

class MaskConditionedAttention(nn.Module):
    def __init__(self, dep_dim, mask_dim, attn_dim):
        super().__init__()
        self.query_proj = nn.Conv2d(dep_dim, attn_dim, 1)
        self.key_proj = nn.Conv2d(mask_dim, attn_dim, 1)
        self.value_proj = nn.Conv2d(mask_dim, attn_dim, 1)
        self.out_proj = nn.Conv2d(attn_dim, dep_dim, 1)

    def forward(self, dep_feat, mask_feat):
        # dep_feat: [B, C, H, W]
        # mask_feat: [B, num_classes, H, W]
        B, C, H, W = dep_feat.shape
        Q = self.query_proj(dep_feat).flatten(2).transpose(1, 2)    # [B, HW, attn_dim]
        K = self.key_proj(mask_feat).flatten(2).transpose(1, 2)     # [B, HW, attn_dim]
        V = self.value_proj(mask_feat).flatten(2).transpose(1, 2)   # [B, HW, attn_dim]
        attn = torch.softmax(torch.bmm(Q, K.transpose(1, 2)) / (K.shape[-1] ** 0.5), dim=-1)  # [B, HW, HW]
        fused = torch.bmm(attn, V)  # [B, HW, attn_dim]
        fused = fused.transpose(1, 2).reshape(B, -1, H, W)          # [B, attn_dim, H, W]
        out = self.out_proj(fused)                                  # [B, C, H, W]
        return out
        
class SEFusionCompressor(nn.Module):
    def __init__(self, in_channels=300, hidden_channels=64):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_channels, hidden_channels // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 8, hidden_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.final = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def forward(self, x):  # [B, 300, H, W]
        x = self.reduce(x)         # [B, 64, H, W]
        attn = self.se(x)          # [B, 64, 1, 1]
        x = x * attn               
        return self.final(x)       # [B, 1, H, W]

class ProgressiveUpsample(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_upsample=3):
        super().__init__()
        layers = []
        channels = in_channels
        for i in range(num_upsample):
            layers.append(nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(channels * 2))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
            channels = channels * 2
        layers.append(nn.Conv2d(channels, out_channels, kernel_size=1))
        self.up = nn.Sequential(*layers)

    def forward(self, x):  # e.g., [B, 1, H, W]
        return self.up(x)  # → [B, 1, H×2ⁿ, W×2ⁿ]

class SegmentationHead(nn.Module):
    def __init__(self, in_channels=300, upsample_times=3):
        super().__init__()
        self.fuse = SEFusionCompressor(in_channels)
        self.upsample = ProgressiveUpsample(in_channels=1, out_channels=1, num_upsample=upsample_times)

    def forward(self, x):  # [B, 300, H, W]
        x = self.fuse(x)         # → [B, 1, H, W]
        x = self.upsample(x)     # → [B, 1, H×2ⁿ, W×2ⁿ]
        return x

class AVSegHead(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes,
        query_num,
        query_generator,
        audio_feat_dim=256,
        embed_dim=256,
        valid_indices=[1, 2, 3],
        scale_factor=4,
        positional_encoding=None,
        use_learnable_queries=True,
        structure_prompt_dim=64,
        decoder_out_dim=16,
        fpn_out_dim=128,
        transformer=None,
        **kwargs
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.query_num = query_num
        self.valid_indices = valid_indices
        self.num_feats = len(valid_indices)
        self.scale_factor = scale_factor
        self.use_learnable_queries = use_learnable_queries

        # 输入投影
        self.in_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, embed_dim, kernel_size=1),
                nn.GroupNorm(32, embed_dim)
            ) for c in in_channels
        ])
        self.fpn = SimpleFPN([embed_dim for _ in valid_indices], fpn_out_dim)

        # 融合模块（多尺度）
        self.mixer = NewCrossModalMixer()

        # Query生成
        self.query_generator = build_generator(**query_generator)
        if use_learnable_queries:
            self.learnable_query = nn.Embedding(query_num, embed_dim)

        # transformer encoder
        self.transformer = build_transformer(**transformer)
        self.level_embed = nn.Parameter(torch.Tensor(self.num_feats, embed_dim))
        nn.init.normal_(self.level_embed)

        if positional_encoding is not None:
            self.positional_encoding = build_positional_encoding(**positional_encoding)
        else:
            self.positional_encoding = None

        # 多尺度解码器
        self.decode_func = SimpleFiLMCrossAttentionDecoder(
            in_dim=fpn_out_dim, audio_dim=audio_feat_dim, structure_dim=structure_prompt_dim,
            out_dim=decoder_out_dim, num_heads=4, num_scales=len(valid_indices)
        )
        
        self.decode_dep = SimpleFiLMCrossAttentionDecoder(
            in_dim=fpn_out_dim, audio_dim=audio_feat_dim, structure_dim=structure_prompt_dim,
            out_dim=decoder_out_dim, num_heads=4, num_scales=len(valid_indices)
        )

        self.func_head = QueryConditionedMaskHead(decoder_out_dim, embed_dim, num_classes)

        self.mask_conditioned_attn = MaskConditionedAttention(
            dep_dim=decoder_out_dim, 
            mask_dim=num_classes, 
            attn_dim=decoder_out_dim 
        )

        self.dep_head  = QueryConditionedMaskHead(decoder_out_dim, embed_dim, num_classes)

        self.learnable_upsample_func = SegmentationHead()
        self.learnable_upsample_dep = SegmentationHead()


    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, feats, audio_feat, audio_feature_list, structure_info=None):
        bs = audio_feat.shape[0]
        # 1. 多尺度输入预处理
        srcs = [self.in_proj[i](feats[i]) for i in self.valid_indices]
        masks = [torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool) for x in srcs]
        pos_embeds = [self.positional_encoding(m) for m in masks] if self.positional_encoding else [0 for _ in masks]

        # 2. transformer encoder
        src_flatten, mask_flatten, lvl_pos_embed_flatten, spatial_shapes = [], [], [], []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            if isinstance(pos_embed, int):
                pos_embed = torch.zeros_like(src)
            else:
                pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
        memory, reference_points = self.transformer.encode(
            src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten
        )

        # 3. 还原回多尺度结构
        mask_feats = []
        start = 0
        for (h, w) in spatial_shapes:
            end = start + h * w
            mask_feats.append(memory[:, start:end, :].transpose(1, 2).reshape(bs, -1, h, w))
            start = end

        # 4. FPN多尺度融合
        mask_feats = self.fpn(mask_feats)  # list of [B, C, H, W]

        # 5. 多尺度视觉-音频融合
        func_v_feats, func_a_feats, dep_v_feats, dep_a_feats = self.mixer(mask_feats, audio_feat)
        # 6. 生成 query
        audio_feat_func = torch.cat([audio_feat] + func_a_feats, dim=1)
        query_func = self.query_generator(audio_feat_func)  # [B, T, 256] -> [B, T, 256]
        audio_feat_dep = torch.cat([audio_feat] + dep_a_feats, dim=1)
        query_dep = self.query_generator(audio_feat_dep)

        if self.use_learnable_queries:
            Q = self.learnable_query.weight.shape[0]
            query_func = query_func + self.learnable_query.weight[None, :Q, :].repeat(bs, 1, 1)
            query_dep = query_dep + self.learnable_query.weight[None, :Q, :].repeat(bs, 1, 1)

        # 7. 多尺度解码
        decoder_out_func = self.decode_func(func_v_feats, func_a_feats, structure_info)
        decoder_out_dep  = self.decode_dep(dep_v_feats,  dep_a_feats,  structure_info)

        # 8. Query-conditioned mask head
        pred_func = self.func_head(decoder_out_func, query_func)  # [B, Q, num_classes, H, W]
        func_mask = pred_func.mean(dim=1)  # [B, num_classes, H, W]

        # === mask作为attention条件 ===
        dep_feat_guided = self.mask_conditioned_attn(decoder_out_dep, func_mask)  # [B, decoder_out_dim, H, W]
        dep_feat_guided = dep_feat_guided + decoder_out_dep  # 残差

        pred_dep  = self.dep_head(dep_feat_guided,  query_dep)   # [B, Q, num_classes, H, W]
        assert pred_func.ndim == 5 and pred_func.shape[2] == 1, f"pred_func shape: {pred_func.shape}"
        pred_func = pred_func.squeeze(2)
        assert pred_dep.ndim == 5 and pred_dep.shape[2] == 1, f"pred_dep shape: {pred_dep.shape}"
        pred_dep = pred_dep.squeeze(2)
        # 9. 聚合query mask为最终mask
        final_func_mask = self.learnable_upsample_func(pred_func)  # [B, num_classes, H, W]
        final_dep_mask  = self.learnable_upsample_dep(pred_dep)
        return final_func_mask, final_dep_mask, pred_func, pred_dep