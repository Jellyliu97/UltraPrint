import torch
import torch.nn as nn
import torch.nn.functional as F

class WaveformEncoder(nn.Module):
    def __init__(self, feature_dim, hidden_dim=64):
        super(WaveformEncoder, self).__init__()
        # Input: (Batch, 1, 3000*T)
        # Target output: (Batch, 10*T, feature_dim)
        
        # Convolutional layers for dimension compression (Downsampling factor: 300)
        # Factorizing 300: 10 * 6 * 5 = 300
        
        # Layer 1: 3000*T -> 300*T (Stride 10)
        self.conv1 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=11, stride=10, padding=5),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        # Layer 2: 300*T -> 50*T (Stride 6)
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=7, stride=6, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        # Layer 3: 50*T -> 10*T (Stride 5)
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=5, stride=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        
        # Layer 4: 10*T -> 10*T (Stride 1) -> Feature projection
        self.conv4 = nn.Sequential(
            nn.Conv1d(256, feature_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU()
        )
        
        # 1D Attention Mechanism for feature fusion
        # Projects (B, T, D) -> (B, T, D) with context
        self.attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=4, batch_first=True)
        
    def forward(self, x):
        # x shape: (B, 1, 3000*T)
        
        x = self.conv1(x) # (B, 64, 300*T)
        x = self.conv2(x) # (B, 128, 50*T)
        x = self.conv3(x) # (B, 256, 10*T)
        x = self.conv4(x) # (B, feature_dim, 10*T)
        
        # Adjust dimensions for Attention: (B, Feature, Time) -> (B, Time, Feature)
        x = x.permute(0, 2, 1)
        
        # Apply Self-Attention
        # Query, Key, Value are all x
        attn_output, _ = self.attention(x, x, x)
        
        return attn_output

class WaveformDecoder(nn.Module):
    def __init__(self, feature_dim):
        super(WaveformDecoder, self).__init__()
        
        # Upsampling layers to reconstruct original waveform
        # Must match Encoder's downsampling in reverse: 300 = 1 * 5 * 6 * 10
        
        # Layer 1: 10T -> 10T (Stride 1)
        # L_out = (L_in - 1)*1 - 2*1 + 3 + op = L_in - 1 - 2 + 3 + op = L_in + op => op=0
        self.tconv1 = nn.ConvTranspose1d(feature_dim, 256, kernel_size=3, stride=1, padding=1, output_padding=0)
        self.bn1 = nn.BatchNorm1d(256)
        
        # Layer 2: 10T -> 50T (Stride 5)
        # 50T = (10T-1)*5 - 2*2 + 5 + op = 50T - 5 - 4 + 5 + op = 50T - 4 + op => op=4
        self.tconv2 = nn.ConvTranspose1d(256, 128, kernel_size=5, stride=5, padding=2, output_padding=4)
        self.bn2 = nn.BatchNorm1d(128)
        
        # Layer 3: 50T -> 300T (Stride 6)
        # 300T = (50T-1)*6 - 2*3 + 7 + op = 300T - 6 - 6 + 7 + op = 300T - 5 + op => op=5
        self.tconv3 = nn.ConvTranspose1d(128, 64, kernel_size=7, stride=6, padding=3, output_padding=5)
        self.bn3 = nn.BatchNorm1d(64)
        
        # Layer 4: 300T -> 3000T (Stride 10)
        # 3000T = (300T-1)*10 - 2*5 + 11 + op = 3000T - 10 - 10 + 11 + op = 3000T - 9 + op => op=9
        self.tconv4 = nn.ConvTranspose1d(64, 32, kernel_size=11, stride=10, padding=5, output_padding=9)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x shape: (B, 10T, feature_dim)
        
        # Permute back to (B, Feature, Time) for Conv1d
        x = x.permute(0, 2, 1)
        
        x = self.relu(self.bn1(self.tconv1(x)))
        x = self.relu(self.bn2(self.tconv2(x)))
        x = self.relu(self.bn3(self.tconv3(x)))
        x = self.tconv4(x) # (B, 32, 3000*T)
        
        return x

class WaveformModel(nn.Module):
    def __init__(self, feature_dim=512):
        super(WaveformModel, self).__init__()
        self.feature_dim = feature_dim
        self.encoder = WaveformEncoder(feature_dim)
        self.decoder = WaveformDecoder(feature_dim)
        
        self.proj_us = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
    def forward(self, x):
        # x: (B, 1, 3000*T)
        
        # Encode -> Intermediate feature [Batch, 10*T, D]
        latent_features = self.encoder(x)
        
        latent_features_pool = torch.mean(latent_features, dim=1)  # Global average pooling over time
        feature_proj = self.proj_us(latent_features_pool) #增加一个投影层，能够防止影响主要的重建任务
        
        # Decode -> Reconstructed waveform
        reconstruction = self.decoder(latent_features)
        
        return latent_features, feature_proj, reconstruction

if __name__ == "__main__":
    # Test the model with dummy data
    T = 5 # 5 seconds
    batch_size = 4
    feature_dim = 512
    frames_per_sec = 10
    
    model = WaveformModel(feature_dim=feature_dim)
    
    # Input: (B, 32, 3000*T)
    input_wav = torch.randn(batch_size, 32, 3000 * T)
    
    latent, _, recon = model(input_wav)
    
    print(f"Input shape: {input_wav.shape}")
    print(f"Latent feature shape: {latent.shape}")
    print(f"Reconstructed shape: {recon.shape}")
    
    expected_latent_shape = (batch_size, T * frames_per_sec, feature_dim)
    expected_recon_shape = input_wav.shape
    
    assert latent.shape == expected_latent_shape, f"Latent shape mismatch! Expected {expected_latent_shape}, got {latent.shape}"
    assert recon.shape == expected_recon_shape, f"Recon shape mismatch! Expected {expected_recon_shape}, got {recon.shape}"
    print("Model test passed!")
