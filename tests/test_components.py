import unittest
import torch
import sys
import os

# Add src to path
sys.path.append(os.path.abspath("src"))

from models.accelsight_net import AccelSightNet, ConvBlock, SpatioTemporalEncoder
from training.losses import MultiTaskLoss

class TestAccelSight(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.num_frames = 5
        self.height, self.width = 128, 128
        self.model = AccelSightNet(num_frames=self.num_frames)
        self.criterion = MultiTaskLoss()

    def test_spatial_encoder_output_shape(self):
        # Test ConvBlock
        block = ConvBlock(3, 16, stride=2)
        x = torch.randn(2, 3, 64, 64)
        out = block(x)
        self.assertEqual(out.shape, (2, 16, 32, 32))

    def test_temporal_encoder_output_shape(self):
        # Test SpatioTemporalEncoder
        # (B, C, T, H, W)
        encoder = SpatioTemporalEncoder(64, 128)
        x = torch.randn(2, 64, 5, 16, 16)
        out = encoder(x)
        self.assertEqual(out.shape, (2, 128, 5, 16, 16))

    def test_full_forward_pass(self):
        x = torch.randn(self.batch_size, self.num_frames, 3, self.height, self.width)
        outputs = self.model(x)
        
        # Grid size at 1/16 resolution
        grid_h, grid_w = self.height // 16, self.width // 16
        
        self.assertEqual(outputs["objectness"].shape, (self.batch_size, 1, grid_h, grid_w))
        self.assertEqual(outputs["bbox"].shape, (self.batch_size, 4, grid_h, grid_w))
        self.assertEqual(outputs["velocity"].shape, (self.batch_size, 3, grid_h, grid_w))
        self.assertEqual(outputs["embedding"].shape, (self.batch_size, 128, grid_h, grid_w))

    def test_loss_calculation(self):
        x = torch.randn(self.batch_size, self.num_frames, 3, self.height, self.width)
        outputs = self.model(x)
        
        grid_h, grid_w = self.height // 16, self.width // 16
        targets = {
            "gt_objectness": torch.zeros(self.batch_size, 1, grid_h, grid_w),
            "gt_bbox": torch.zeros(self.batch_size, 4, grid_h, grid_w),
            "gt_velocity": torch.zeros(self.batch_size, 3, grid_h, grid_w),
            "gt_ids": torch.zeros(self.batch_size, grid_h, grid_w)
        }
        
        # Put one object in
        targets["gt_objectness"][0, 0, 4, 4] = 1.0
        
        losses = self.criterion(outputs, targets)
        self.assertIn("total_loss", losses)
        self.assertTrue(losses["total_loss"] > 0)
        self.assertFalse(torch.isnan(losses["total_loss"]))

if __name__ == "__main__":
    unittest.main()
