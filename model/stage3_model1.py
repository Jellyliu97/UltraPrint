import torch
import torch.nn as nn
import sys
import os

# Add the project root to sys.path to allow importing from model.stage1_model1
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from model.stage1_model1 import WaveformEncoder
except ImportError:
    # Fallback if running from within model directory
    from stage1_model1 import WaveformEncoder

class CrossAttention(nn.Module):
    def __init__(self, feature_dim, num_heads=8, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        # query, key, value shape: (Batch, Time, Feature)
        attn_output, _ = self.multihead_attn(query, key, value)
        # Residual connection + Norm
        # The user requested: "Combine with original V and U features"
        # Standard Pre-Norm or Post-Norm Transformer block behavior satisfies this (x + sublayer(x))
        output = self.norm(query + self.dropout(attn_output))
        return output

# class U2VModel(nn.Module):
#     def __init__(self, feature_dim=512, hidden_dim=256):
#         super(U2VModel, self).__init__()
#         self.model = nn.Sequential(
#             nn.Linear(feature_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(hidden_dim, feature_dim)
#         )
        
#     def forward(self, x):
#         return self.model(x)
    
# class V2UModel(nn.Module):
#     def __init__(self, feature_dim=512, hidden_dim=256):
#         super(V2UModel, self).__init__()
#         self.model = nn.Sequential(
#             nn.Linear(feature_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(hidden_dim, feature_dim)
#         )
        
#     def forward(self, x):
#         return self.model(x)

class FeatureFusion(nn.Module):
    """
    Multi-layer MLP for User Verification.
    """
    def __init__(self, input_dim, hidden_dim=256, output_dim=256):
        super(FeatureFusion, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim)
        )

    def forward(self, x):
        return self.mlp(x)

class Stage3Model(nn.Module):
    def __init__(self, feature_dim=512, hidden_dim=64):
        super(Stage3Model, self).__init__()
        
        # 1. Waveform Encoder (from stage 1)
        # Initialize WaveformEncoder
        self.encoder = WaveformEncoder(feature_dim, hidden_dim)
        
        # 2. Cross Attention Modules
        # U: Waveform features, V: Image features
        # U attends to V (Query=U, Key=V, Value=V)
        self.dim_reduction_v = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, feature_dim // 2),
        )
        self.dim_reduction_u = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, feature_dim // 2),
        )
            
        self.cross_attn_alfa = CrossAttention(feature_dim // 2) # U attending V
        
        # V attends to U (Query=V, Key=U, Value=U)
        self.cross_attn_beta = CrossAttention(feature_dim // 2) # V attending U
        
        # 3. Aggregation/Fusion
        # Concatenate the outputs from both attention branches
        self.feature_convert = nn.Sequential(
            nn.Linear(feature_dim, feature_dim //2),
            nn.ReLU(),  
            nn.Linear(feature_dim // 2, feature_dim //2),
        )
        
        # 4. Classifier
        self.feature_fusion = FeatureFusion(feature_dim // 2)
        
        
        # self.model_u2v = U2VModel(feature_dim)  # Model to reconstruct video features from us features
        # self.model_v2u = V2UModel(feature_dim)  # Model to reconstruct

    def forward(self, waveform, image_features):
        """
        waveform: (Batch, 32, 3000*T) - Input raw waveform
        image_features (V): (Batch, 10*T, feature_dim) - Encoded image features
        """
        
        # Extract features U from WaveformEncoder
        # U shape: (Batch, 10*T, feature_dim)
        u = self.encoder(waveform) #[B, 10*T, 512]
        v = image_features #[B, 10*T, 512]
        
        # Dimensionality Reduction
        u = self.dim_reduction_u(u)  #[B, 10*T, 256]
        v = self.dim_reduction_v(v)  #[B, 10*T, 256]
        
        u_pooled = torch.mean(u, dim=1)  #[B, 256]
        v_pooled = torch.mean(v, dim=1)  #[B, 256]
        
        # Cross Attention
        # Side A: U attends to V (Query=U, Key=V, Value=V)
        # Output is enriched U features
        u_prime = self.cross_attn_alfa(query=u, key=v, value=v)
        
        # Side B: V attends to U (Query=V, Key=U, Value=U)
        # Output is enriched V features
        v_prime = self.cross_attn_beta(query=v, key=u, value=u)
        
        # Feature Fusion
        # Concatenate along feature dimension: (Batch, 10*T, 2*feature_dim)
        combined = torch.cat((u_prime, v_prime), dim=-1)
        
        # Project back to feature_dim
        fused = self.feature_convert(combined) # (Batch, 10*T, feature_dim) -> (Batch, 10*T, feature_dim//2)
        
        # Classification
        # Global Average Pooling over time dimension to get a single vector per sample
        pooled_features = torch.mean(fused, dim=1) # (Batch, feature_dim//2)
        
        class_features = pooled_features + u_pooled + v_pooled #直接相加
        # class_features = torch.cat((pooled_features, u_pooled, v_pooled), dim=-1) #cat
        # class_features = u_pooled 
        
        # class_features = v_pooled  ###test
        
        output_feature = self.feature_fusion(class_features) # (Batch, 1)
        # output_feature = class_features # (Batch, 1) ###test
        
        return v_pooled, u_pooled, output_feature  #vedio feature, fusion feature

    def freeze_model(self):
        """Freezes the parameters of the WaveformEncoder."""
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        # for param in self.model_u2v.parameters():
        #     param.requires_grad = False
        
        # for param in self.model_v2u.parameters():
        #     param.requires_grad = False
        
        print("Waveform Encoder and reconstruction models frozen.")

if __name__ == "__main__":
    # Test the model
    T = 5
    batch_size = 4
    feature_dim = 512
    frames_per_sec = 10
    
    # Initialize model
    model = Stage3Model(feature_dim=feature_dim)
    
    # Test freezing
    model.freeze_waveform_encoder()
    print("Waveform encoder frozen.")
    
    # Dummy Inputs
    input_wav = torch.randn(batch_size, 32, 3000 * T) # 32 channels based on stage1 code
    input_img_features = torch.randn(batch_size, frames_per_sec * T, feature_dim)
    
    v_pooled, u_pooled, output_feature = model(input_wav, input_img_features)
    
    print(f"Input Waveform: {input_wav.shape}")
    print(f"Input Image Features: {input_img_features.shape}")
    print(f"Pooled Video Feature Shape: {v_pooled.shape}")
    print(f"Pooled Waveform Feature Shape: {u_pooled.shape}")
    print(f"Output Fusion Feature Shape: {output_feature.shape}")
    
    print("Stage2 Model test passed!")
