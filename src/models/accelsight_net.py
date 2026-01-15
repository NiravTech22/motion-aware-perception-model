import torch
import torch.nn as nn
import torch.nn.functional as F

from .spatial_encoder import SpatialEncoder
from .temporal_encoder import SpatioTemporalEncoder
from .lite_head import LiteDetectionHead, LiteRegressionHead
from .control_head import ControlHead
from .post_processor import PostProcessor

class AccelSightNet(nn.Module):
    """
    AccelSight Spatiotemporal Network.
    Integrates modular spatial and temporal encoders with multi-task prediction heads,
    now including a learning-based control head.
    """
    def __init__(self, num_frames=5, input_channels=3, base_channels=16, embedding_dim=128, control_dim=3):
        super(AccelSightNet, self).__init__()
        self.num_frames = num_frames
        
        # --- Stage A: Spatial Encoder ---
        self.spatial_encoder = SpatialEncoder(in_channels=input_channels, base_channels=base_channels)
        s_out = self.spatial_encoder.out_channels # e.g., 128
        
        # --- Stage B: Temporal Encoder ---
        self.temporal_encoder = SpatioTemporalEncoder(in_channels=s_out, out_channels=s_out * 2)
        t_out = s_out * 2 # e.g., 256
        
        # Bottleneck to collapse temporal dimension and further compress
        self.bottleneck = nn.Sequential(
            nn.Conv2d(t_out, t_out, kernel_size=3, padding=1, stride=2, bias=False), # 1/16 resolution
            nn.BatchNorm2d(t_out),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        # --- Stage C: Detection Head ---
        self.det_head = LiteDetectionHead(t_out)
        
        # --- Stage D: Velocity Head ---
        self.vel_head = LiteRegressionHead(t_out, out_dim=3)
        
        # --- Stage E: Embedding Head ---
        self.embed_head = LiteRegressionHead(t_out, out_dim=embedding_dim)

        # --- Stage F: Control Head ---
        self.control_head = ControlHead(t_out, control_dim=control_dim)

    def forward(self, x):
        """
        Input x: (B, T, C, H, W)
        """
        B, T, C, H, W = x.shape
        
        # 1. Apply Spatial Encoding to each frame independently
        # (B*T, C, H, W) -> (B*T, S_OUT, H/8, W/8)
        x_flat = x.view(B * T, C, H, W)
        spatial_feats = self.spatial_encoder(x_flat)
        
        # 2. Reshape for Temporal Encoder
        # (B, S_OUT, T, H/8, W/8)
        _, Sf, Sh, Sw = spatial_feats.shape
        temporal_in = spatial_feats.view(B, T, Sf, Sh, Sw).permute(0, 2, 1, 3, 4)
        
        # 3. Apply Temporal Encoding
        temporal_feats = self.temporal_encoder(temporal_in) # (B, T_OUT, T, H/8, W/8)
        
        # 4. Collapse Time (Max pool over temporal dim)
        collapsed_feat = torch.max(temporal_feats, dim=2)[0] # (B, T_OUT, H/8, W/8)
        
        # 5. Bottleneck processing
        feat = self.bottleneck(collapsed_feat) # (B, T_OUT, H/16, W/16)
        
        # 6. Multi-head predictions
        objectness, bbox = self.det_head(feat)
        velocity = self.vel_head(feat)
        embedding = self.embed_head(feat)
        controls = self.control_head(feat)
        
        # Normalize embedding
        embedding = F.normalize(embedding, p=2, dim=1)
        
        return {
            "objectness": objectness,
            "bbox": bbox,
            "velocity": velocity,
            "embedding": embedding,
            "controls": controls
        }

if __name__ == "__main__":
    # Test input
    model = AccelSightNet(num_frames=5)
    dummy = torch.randn(1, 5, 3, 256, 256)
    out = model(dummy)
    print("Full AccelSightNet Forward Pass Successful.")
    for k, v in out.items():
        print(f"{k} shape: {v.shape}")
