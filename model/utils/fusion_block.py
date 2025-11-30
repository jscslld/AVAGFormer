import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class SC2FusionBlock(nn.Module):
    """
    Cross-Attention Fusion Block
    """
    def __init__(self, img_dim=256, aud_dim=128, n_heads=8, dropout=0.):
        super().__init__()
        self.img_dim = img_dim
        self.aud_dim = aud_dim
        self.n_heads = n_heads
        self.scale_img = (img_dim // n_heads) ** -0.5
        self.scale_aud = (aud_dim // n_heads) ** -0.5

        # vision: Q/K/V/Out
        self.q_proj_image = nn.Linear(img_dim, img_dim)
        self.k_proj_image = nn.Linear(img_dim, img_dim)
        self.v_proj_image = nn.Linear(img_dim, img_dim)
        self.out_proj_image = nn.Linear(aud_dim, img_dim)  # 融合后输出到img_dim

        # audio: Q/K/V/Out
        self.q_proj_audio = nn.Linear(aud_dim, aud_dim)
        self.k_proj_audio = nn.Linear(aud_dim, aud_dim)
        self.v_proj_audio = nn.Linear(aud_dim, aud_dim)
        self.out_proj_audio = nn.Linear(img_dim, aud_dim)  # 融合后输出到aud_dim

        # 条件投影
        self.cond_proj_img = nn.Sequential(
            nn.Linear(img_dim + 2, img_dim),
            nn.ReLU(),
            nn.Linear(img_dim, n_heads)
        )
        self.cond_proj_aud = nn.Sequential(
            nn.Linear(aud_dim + 2, aud_dim),
            nn.ReLU(),
            nn.Linear(aud_dim, n_heads)
        )

        # 门控
        self.gate_fc_img = nn.Sequential(
            nn.Linear(img_dim, img_dim),
            nn.Sigmoid()
        )
        self.gate_fc_aud = nn.Sequential(
            nn.Linear(aud_dim, aud_dim),
            nn.Sigmoid()
        )

    def forward(self, image_feat, audio_feat, task_prompt=None, spatial_mask=None, time_mask=None):
        B, N_v, C_img = image_feat.shape
        B, N_a, C_aud = audio_feat.shape

        # task_prompt: B x C_img, B x C_aud
        if task_prompt is None:
            task_prompt_img = image_feat.mean(dim=1)  # B x C_img
            task_prompt_aud = audio_feat.mean(dim=1)  # B x C_aud
        elif isinstance(task_prompt, (tuple, list)):
            task_prompt_img, task_prompt_aud = task_prompt
        else:
            task_prompt_img = task_prompt
            task_prompt_aud = task_prompt

        # 条件
        spatial_cond = spatial_mask.mean(dim=1) if spatial_mask is not None else image_feat.norm(dim=-1).mean(dim=1, keepdim=True)  # [B,1]
        time_cond = time_mask if time_mask is not None else audio_feat.norm(dim=-1).mean(dim=1, keepdim=True)  # [B,1]

        # --------- Audio attending to vision ---------
        Q_audio = self.q_proj_audio(audio_feat).view(B, N_a, self.n_heads, C_aud // self.n_heads).transpose(1, 2)  # [B, H, N_a, d_a]
        K_image = self.k_proj_image(image_feat).view(B, N_v, self.n_heads, C_img // self.n_heads).transpose(1, 2)  # [B, H, N_v, d_i]
        V_image = self.v_proj_image(image_feat).view(B, N_v, self.n_heads, C_img // self.n_heads).transpose(1, 2)  # [B, H, N_v, d_i]

        # condition bias
        cond_aud = torch.cat([task_prompt_aud, spatial_cond, time_cond], dim=-1)
        cond_bias_aud = self.cond_proj_aud(cond_aud).unsqueeze(2).unsqueeze(3)  # [B, H, 1, 1]

        # cross-attention
        attn_audio = (Q_audio @ K_image.transpose(-2, -1)) * self.scale_aud + cond_bias_aud  # [B, H, N_a, N_v]
        attn_audio = F.softmax(attn_audio, dim=-1)
        # 融合
        fused_audio = (attn_audio @ V_image)  # [B, H, N_a, d_i]
        fused_audio = fused_audio.transpose(1, 2).contiguous().view(B, N_a, C_img)  # [B, N_a, C_img]
        fused_audio = self.out_proj_audio(fused_audio)  # [B, N_a, C_aud]
        gate_audio = self.gate_fc_aud(task_prompt_aud).unsqueeze(1)  # [B,1,C_aud]
        audio_feat = gate_audio * fused_audio + (1 - gate_audio) * audio_feat

        # --------- Vision attending to audio ---------
        Q_image = self.q_proj_image(image_feat).view(B, N_v, self.n_heads, C_img // self.n_heads).transpose(1, 2)  # [B, H, N_v, d_i]
        K_audio = self.k_proj_audio(audio_feat).view(B, N_a, self.n_heads, C_aud // self.n_heads).transpose(1, 2)  # [B, H, N_a, d_a]
        V_audio = self.v_proj_audio(audio_feat).view(B, N_a, self.n_heads, C_aud // self.n_heads).transpose(1, 2)  # [B, H, N_a, d_a]

        cond_img = torch.cat([task_prompt_img, spatial_cond, time_cond], dim=-1)
        cond_bias_img = self.cond_proj_img(cond_img).unsqueeze(2).unsqueeze(3)  # [B, H, 1, 1]

        attn_image = (Q_image @ K_audio.transpose(-2, -1)) * self.scale_img + cond_bias_img  # [B, H, N_v, N_a]
        attn_image = F.softmax(attn_image, dim=-1)
        fused_image = (attn_image @ V_audio)  # [B, H, N_v, d_a]
        fused_image = fused_image.transpose(1, 2).contiguous().view(B, N_v, C_aud)  # [B, N_v, C_aud]
        fused_image = self.out_proj_image(fused_image)  # [B, N_v, C_img]
        gate_image = self.gate_fc_img(task_prompt_img).unsqueeze(1)  # [B,1,C_img]
        image_feat = gate_image * fused_image + (1 - gate_image) * image_feat

        return image_feat, audio_feat
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class NewCrossModalMixer(nn.Module):
    def __init__(self, num_scales=3, vision_dim=128, audio_dim=256, n_heads=8, num_layers=1, dropout=0.):
        super().__init__()
        self.audio_dim = audio_dim
        self.vision_proj_in = nn.ModuleList([
            nn.Linear(vision_dim, audio_dim) for _ in range(num_scales)
        ])
        self.vision_proj_out = nn.ModuleList([
            nn.Linear(audio_dim, vision_dim) for _ in range(num_scales)
        ])
        self.fusion = nn.ModuleList([
            SC2FusionBlock(img_dim=audio_dim, aud_dim=audio_dim, n_heads=n_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.task_prompt_img = nn.Embedding(2, audio_dim)
        self.task_prompt_aud = nn.Embedding(2, audio_dim)

    def forward(self, vision_feats, audio_feat):
        """
        vision_feats: List[Tensor], each is [B, 128, H, W]
        audio_feat: [B, 20, 256]
        """
        results = [[], [], [], []]  # [func_v, func_a, dep_v, dep_a]

        if audio_feat.dim() == 2:
            audio_feat = audio_feat.unsqueeze(1)  # [B, 1, C_aud]

        for idx, vision_feat in enumerate(vision_feats):
            bs, c_img, h, w = vision_feat.shape
            N_v = h * w
            vision_feat_flat = vision_feat.flatten(2).transpose(1, 2)  # [B, N_v, 128]
            # 1. 投影到audio_dim
            vision_feat_proj = self.vision_proj_in[idx](vision_feat_flat)  # [B, N_v, 256]
            spatial_mask = vision_feat_proj.norm(dim=-1, keepdim=True)  # [B, N_v, 1]
            time_mask = audio_feat.norm(dim=-1).mean(dim=1, keepdim=True)  # [B, 1]

            for task_id in range(2):
                task_prompt_img = self.task_prompt_img.weight[task_id].unsqueeze(0).expand(bs, -1)  # [B, 256]
                task_prompt_aud = self.task_prompt_aud.weight[task_id].unsqueeze(0).expand(bs, -1)  # [B, 256]
                v_feat, a_feat = vision_feat_proj, audio_feat
                for block in self.fusion:
                    v_feat, a_feat = block(
                        v_feat, a_feat,
                        task_prompt=(task_prompt_img, task_prompt_aud),
                        spatial_mask=spatial_mask,
                        time_mask=time_mask
                    )
                # 2. 融合后投影回128通道
                v_feat_out = self.vision_proj_out[idx](v_feat)  # [B, N_v, 128]
                v_feat_img = v_feat_out.transpose(1, 2).reshape(bs, c_img, h, w)
                if task_id == 0:
                    results[0].append(v_feat_img)  # func_v
                    results[1].append(a_feat)      # func_a
                else:
                    results[2].append(v_feat_img)  # dep_v
                    results[3].append(a_feat)      # dep_a

        return results[0], results[1], results[2], results[3]




def build_fusion_block(type, **kwargs):
    if type == 'NewCrossModalMixer':
        return NewCrossModalMixer(**kwargs)
    else:
        raise ValueError
