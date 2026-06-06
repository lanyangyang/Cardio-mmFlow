import torch
import torch.nn as nn
import torch.nn.functional as F

class ResnetBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels=None, dropout=0.0):
        super().__init__()
        out_channels = out_channels or in_channels
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect')
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect')
        self.dropout = nn.Dropout(dropout)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.shortcut(x)

class Downsample1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Stride 2 convolution for downsampling
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class Upsample1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        return self.conv(x)

class SignalVAE(nn.Module):
    def __init__(
        self, 
        length=1024,            # Input signal length
        in_channels=1,          # Input channels (ECG=1)
        base_channels=64,       # Base hidden channels
        ch_mult=(1, 2, 4, 4),   # Channel multipliers (4 layers = 16x downsample)
        latent_dim=4,           # Latent dimension channels
        dropout=0.0
    ):
        super().__init__()
        
        self.embed_dim = latent_dim
        
        # --- Encoder ---
        self.encoder_layers = nn.ModuleList()
        curr_channels = base_channels
        self.encoder_in = nn.Conv1d(in_channels, curr_channels, kernel_size=3, padding=1)
        
        # Downsampling path
        for i, mult in enumerate(ch_mult):
            out_channels = base_channels * mult
            self.encoder_layers.append(nn.Sequential(
                ResnetBlock1D(curr_channels, out_channels, dropout),
                ResnetBlock1D(out_channels, out_channels, dropout)
            ))
            if i != len(ch_mult) - 1:
                self.encoder_layers.append(Downsample1D(out_channels))
            curr_channels = out_channels
            
        # Mid Block
        self.mid_block_enc = nn.Sequential(
            ResnetBlock1D(curr_channels, curr_channels, dropout),
            ResnetBlock1D(curr_channels, curr_channels, dropout),
        )
        
        self.norm_out = nn.GroupNorm(8, curr_channels)
        self.conv_out = nn.Conv1d(curr_channels, 2 * latent_dim, kernel_size=3, padding=1)
        
        # --- Decoder ---
        self.decoder_in = nn.Conv1d(latent_dim, curr_channels, kernel_size=3, padding=1)
        
        self.mid_block_dec = nn.Sequential(
            ResnetBlock1D(curr_channels, curr_channels, dropout),
            ResnetBlock1D(curr_channels, curr_channels, dropout),
        )
        
        self.decoder_layers = nn.ModuleList()
        reversed_mult = list(reversed(ch_mult))
        
        for i, mult in enumerate(reversed_mult):
            out_channels = base_channels * mult
            self.decoder_layers.append(nn.Sequential(
                ResnetBlock1D(curr_channels, out_channels, dropout),
                ResnetBlock1D(out_channels, out_channels, dropout)
            ))
            if i != len(ch_mult) - 1:
                self.decoder_layers.append(Upsample1D(out_channels))
            curr_channels = out_channels
            
        self.norm_dec = nn.GroupNorm(8, curr_channels)
        self.conv_dec = nn.Conv1d(curr_channels, in_channels, kernel_size=3, padding=1)

    def encode(self, x):
        h = self.encoder_in(x)
        for layer in self.encoder_layers:
            h = layer(h)
        h = self.mid_block_enc(h)
        h = self.norm_out(h)
        h = F.silu(h)
        moments = self.conv_out(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        return mean, logvar

    def decode(self, z):
        h = self.decoder_in(z)
        h = self.mid_block_dec(h)
        for layer in self.decoder_layers:
            h = layer(h)
        h = self.norm_dec(h)
        h = F.silu(h)
        x_recon = self.conv_dec(h)
        x_recon = torch.tanh(x_recon)
        return x_recon

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_recon = self.decode(z)
        return x_recon, mean, logvar