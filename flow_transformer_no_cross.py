import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def modulate(x, shift, scale):
    return x + shift.unsqueeze(1) + scale.unsqueeze(1)

def get_1d_sincos_pos_embed(embed_dim, length):
    assert embed_dim % 2 == 0, "embed_dim must be even"
    
    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    
    div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float32) * 
                        -(math.log(10000.0) / embed_dim))
    
    pos_embed = torch.zeros(length, embed_dim)
    pos_embed[:, 0::2] = torch.sin(position * div_term)
    pos_embed[:, 1::2] = torch.cos(position * div_term)
    
    return pos_embed.unsqueeze(0)

class GaussianFourierProjection(nn.Module):
    def __init__(self, embed_dim=64, scale=1.0):
        super().__init__()
        self.register_buffer('W', torch.randn(embed_dim // 2, 1) * scale)
        self.embed_dim = embed_dim
        
    def forward(self, x):
        x_proj = 2 * torch.pi * x @ self.W.T
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class PhysicsMetaProjector(nn.Module):
    def __init__(self, output_dim=256):
        super().__init__()
        
        self.register_buffer('bmi_bounds', torch.tensor([13.0, 35.0]))
        self.register_buffer('age_bounds', torch.tensor([15.0, 70.0]))
        self.register_buffer('h_bounds',   torch.tensor([140.0, 210.0]))
        self.register_buffer('w_bounds',   torch.tensor([30.0, 120.0]))
        
        self.cont_encoder = GaussianFourierProjection(embed_dim=8, scale=10.0)
        self.gender_encoder = nn.Embedding(2, 8)
        
        self.mlp = nn.Sequential(
            nn.Linear(40, 64),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        nn.init.normal_(self.mlp[-1].weight, std=0.01)
        nn.init.zeros_(self.mlp[-1].bias)

    def normalize(self, tensor, bounds):
        min_v, max_v = bounds
        tensor = torch.clamp(tensor, min_v, max_v)
        return (tensor - min_v) / (max_v - min_v)

    def forward(self, raw_meta):
        B = raw_meta.shape[0]
        device = raw_meta.device
        dtype = raw_meta.dtype
        
        is_cfg_sample = torch.isnan(raw_meta[:, 0])
        
        if is_cfg_sample.all():
            output_dim = self.mlp[-1].weight.shape[0]
            return torch.zeros(B, output_dim, device=device, dtype=dtype)

        raw_meta = raw_meta.clone()
        if is_cfg_sample.any():
            raw_meta[is_cfg_sample] = torch.tensor([0.0, 40.0, 170.0, 70.0, 24.0], 
                                                    device=device, dtype=dtype)

        gender = torch.clamp(raw_meta[:, 0], 0, 1).long()
        age_norm = self.normalize(raw_meta[:, 1:2], self.age_bounds)
        h_norm   = self.normalize(raw_meta[:, 2:3], self.h_bounds)
        w_norm   = self.normalize(raw_meta[:, 3:4], self.w_bounds)
        bmi_norm = self.normalize(raw_meta[:, 4:5], self.bmi_bounds)

        z_gender = self.gender_encoder(gender)  # [B, 8]
        z_age = self.cont_encoder(age_norm)     # [B, 8]
        z_h   = self.cont_encoder(h_norm)       # [B, 8]
        z_w   = self.cont_encoder(w_norm)       # [B, 8]
        z_bmi = self.cont_encoder(bmi_norm)     # [B, 8]

        z_in = torch.cat([z_gender, z_age, z_h, z_w, z_bmi], dim=-1)  # [B, 40]
        meta_emb = self.mlp(z_in)  # [B, output_dim]
        
        if is_cfg_sample.any():
            meta_emb[is_cfg_sample] = 0.0
        
        return meta_emb

class FLiM(nn.Module):
    def __init__(self, hidden_size, condition_size=None):
        super().__init__()
        if condition_size is None:
            condition_size = hidden_size
        
        self.norm = nn.LayerNorm(hidden_size)
        
        self.scale_proj = nn.Linear(hidden_size + condition_size, hidden_size)
        self.shift_proj = nn.Linear(hidden_size, hidden_size)
        
        nn.init.normal_(self.scale_proj.weight, std=0.02)
        nn.init.zeros_(self.scale_proj.bias)
        nn.init.zeros_(self.shift_proj.weight)
        nn.init.zeros_(self.shift_proj.bias)

    def forward(self, x, time_emb, cond_emb):
        combined = torch.cat([time_emb, cond_emb], dim=-1)
        scale_raw = self.scale_proj(combined)
        scale_raw = torch.clamp(scale_raw, min=-10.0, max=10.0)
        damping = torch.nn.functional.softplus(scale_raw)
        damping = torch.clamp(damping, max=5.0)
        scale = torch.exp(-damping)
        scale = torch.clamp(scale, min=0.01, max=1.0)
        
        shift_raw = self.shift_proj(time_emb)
        shift_raw = torch.clamp(shift_raw, min=-10.0, max=10.0)
        shift = shift_raw * 0.1
        shift = torch.clamp(shift, min=-1.0, max=1.0)
        
        normed_x = self.norm(x)
        result = normed_x * scale.unsqueeze(1) + shift.unsqueeze(1)
        return torch.clamp(result, -10.0, 10.0)

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t):
        half_dim = self.frequency_embedding_size // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        token_embeddings = self.mlp(emb)
        return token_embeddings

class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, use_flim=True, dropout=0.1):
        super().__init__()
        self.use_flim = use_flim
        
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True, dropout=dropout)
        self.attn_dropout = nn.Dropout(dropout)
        
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_size)
        )
        self.mlp_dropout = nn.Dropout(dropout)
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        
        if self.use_flim:
            self.flim_attn = FLiM(hidden_size)
            self.flim_mlp = FLiM(hidden_size)

    def forward(self, x, c, cond=None):
        if cond is None:
            cond = c
        
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        
        shift_msa = torch.clamp(shift_msa, -10.0, 10.0)
        scale_msa = torch.clamp(scale_msa, -5.0, 5.0)
        gate_msa = torch.clamp(gate_msa, -10.0, 10.0)
        shift_mlp = torch.clamp(shift_mlp, -10.0, 10.0)
        scale_mlp = torch.clamp(scale_mlp, -5.0, 5.0)
        gate_mlp = torch.clamp(gate_mlp, -10.0, 10.0)
        
        attn_input = self.norm1(x)
        if self.use_flim:
            attn_input = self.flim_attn(attn_input, c, cond)
        attn_input = modulate(attn_input, shift_msa, scale_msa)
        
        attn_output = self.attn(attn_input, attn_input, attn_input)[0]
        
        if torch.isnan(attn_output).any() or torch.isinf(attn_output).any():
            attn_output = torch.nan_to_num(attn_output, nan=0.0, posinf=1.0, neginf=-1.0)
        
        attn_output = self.attn_dropout(attn_output)
        x = x + gate_msa.unsqueeze(1) * attn_output
        
        mlp_input = self.norm2(x)
        if self.use_flim:
            mlp_input = self.flim_mlp(mlp_input, c, cond)
        mlp_input = modulate(mlp_input, shift_mlp, scale_mlp)
        
        mlp_output = self.mlp(mlp_input)
        
        if torch.isnan(mlp_output).any() or torch.isinf(mlp_output).any():
            mlp_output = torch.nan_to_num(mlp_output, nan=0.0, posinf=1.0, neginf=-1.0)
        
        mlp_output = self.mlp_dropout(mlp_output)
        x = x + gate_mlp.unsqueeze(1) * mlp_output
        
        x = torch.clamp(x, -100.0, 100.0)
        
        return x

class DirectFlowTransformer(nn.Module):
    def __init__(
        self, 
        in_channels=4,
        seq_len=64,
        hidden_size=512,
        depth=12,
        num_heads=8,
        mlp_ratio=4.0,
        use_flim=True,
        use_metadata=False,
        dropout=0.1
    ):
        super().__init__()
        self.in_channels = in_channels
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.use_flim = use_flim
        self.use_metadata = use_metadata
        
        self.x_embed = nn.Linear(in_channels, hidden_size)
        
        pos_embed = get_1d_sincos_pos_embed(hidden_size, seq_len)
        self.register_buffer('pos_embed', pos_embed)
        
        self.t_embedder = TimestepEmbedder(hidden_size)
        
        if self.use_metadata:
            self.meta_projector = PhysicsMetaProjector(output_dim=hidden_size)
        
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, use_flim=use_flim, dropout=dropout) 
            for _ in range(depth)
        ])
        
        self.final_layer = nn.ModuleList([
            nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6),
            nn.Linear(hidden_size, in_channels, bias=True)
        ])
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        
        self.initialize_weights()

    def initialize_weights(self):
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer[-1].weight, 0)
        nn.init.constant_(self.final_layer[-1].bias, 0)

    def forward(self, x, t, meta=None):
        x_input = x.transpose(1, 2)
        
        x_emb = self.x_embed(x_input) + self.pos_embed
        t_emb = self.t_embedder(t)
        
        if self.use_metadata and meta is not None:
            meta_emb = self.meta_projector(meta)
        else:
            meta_emb = None
        
        if self.use_flim and self.use_metadata and meta_emb is not None:
            cond_emb = meta_emb
        else:
            cond_emb = None
        
        x_out = x_emb
        for block in self.blocks:
            x_out = block(x_out, t_emb, cond_emb)
            
        shift, scale = self.adaLN_modulation(t_emb).chunk(2, dim=1)
        x_out = modulate(self.final_layer[0](x_out), shift, scale)
        x_out = self.final_layer[1](x_out)
        
        return x_out.transpose(1, 2)