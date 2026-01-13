import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """spatial encoding using 2d convolutions."""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class SpatioTemporalEncoder(nn.Module):

    """temporal encoding and 3d convolutions using frame stack."""
    def __init__(self, in_channels, out_channels):
        super(SpatioTemporalEncoder, self).__init__()

        #input shape: (Batch, Channels, Depth/Frames, Height, Width)
        self.conv3d = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.conv3d(x)

class AccelSightNet(nn.Module):
    def __init__(self, num_frames=5, input_channels=3, embedding_dim=128):
        super(AccelSightNet, self).__init__()
        self.num_frames = num_frames
        

        self.spatial_feat = nn.Sequential(
            ConvBlock(input_channels, 16, stride=2), # 1/2
            ConvBlock(16, 32, stride=2),            # 1/4
            ConvBlock(32, 64, stride=2),            # 1/8
        )
        
        self.temporal_encoder = SpatioTemporalEncoder(64, 128)
        
        # urther downsampling stage
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2, bias=False), # 1/16
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        # --- Stage C: Detection Head ---
        # Outputs objectness heatmaps and bbox offsets
        self.det_head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            # Objectness, BBox (x, y, w, h)
            nn.Conv2d(128, 1 + 4, kernel_size=1) 
        )
        
        # --- Stage D: Velocity Head ---
        # Regress per-object velocity vector (vx, vy, vz)
        self.vel_head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 3, kernel_size=1)
        )
        
        # --- Stage E: Embedding Head ---
        # Produce fixed-length feature vector for tracking
        self.embed_head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, embedding_dim, kernel_size=1)
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, T, C, H, W) where T is frames
        Returns:
            Dictionary of outputs: objectness, bbox, velocity, embedding
        """
        B, T, C, H, W = x.shape
        # Apply spatial encoder to each frame
        # (B*T, C, H, W) -> (B*T, 64, H/8, W/8)
        x = x.view(B * T, C, H, W)
        spatial_out = self.spatial_feat(x)
        
        # Reshape for temporal encoder
        # (B, 64, T, H/8, W/8)
        _, fC, fH, fW = spatial_out.shape
        temp_in = spatial_out.view(B, T, fC, fH, fW).permute(0, 2, 1, 3, 4)
        
        # Stage B: Temporal Encoding
        temp_out = self.temporal_encoder(temp_in) # (B, 128, T, H/8, W/8)
        
        # Pool/Collapse temporal dimension (e.g., take last frame or mean)
        # Here we use max pooling over time to capture the most significant features
        feat = torch.max(temp_out, dim=2)[0] # (B, 128, H/8, W/8)
        
        # Bottleneck downsampling
        feat = self.bottleneck(feat) # (B, 256, H/16, W/16)
        
        # Multi-head outputs
        objectness_bbox = self.det_head(feat)
        objectness = objectness_bbox[:, :1, :, :]
        bbox = objectness_bbox[:, 1:, :, :]
        
        velocity = self.vel_head(feat)
        embedding = self.embed_head(feat)
        
        # Normalize embeddings for cosine similarity
        embedding = F.normalize(embedding, p=2, dim=1)
        
        return {
            "objectness": objectness,
            "bbox": bbox,
            "velocity": velocity,
            "embedding": embedding
        }

if __name__ == "__main__":
    # Test tensor shapes
    model = AccelSightNet(num_frames=5)
    dummy_input = torch.randn(2, 5, 3, 256, 256) # Batch=2, Frames=5, RGB, 256x256
    outputs = model(dummy_input)
    
    print("Input shape:", dummy_input.shape)
    for k, v in outputs.items():
        print(f"{k} output shape: {v.shape}")
