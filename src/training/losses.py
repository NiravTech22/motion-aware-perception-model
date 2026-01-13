import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskLoss(nn.Module):
    def __init__(self, w_obj=1.0, w_bbox=5.0, w_vel=2.0, w_embed=1.0):
        super(MultiTaskLoss, self).__init__()
        self.w_obj = w_obj
        self.w_bbox = w_bbox
        self.w_vel = w_vel
        self.w_embed = w_embed
        
        # Loss components
        self.obj_loss = nn.BCEWithLogitsLoss()
        self.bbox_loss = nn.SmoothL1Loss()
        self.vel_loss = nn.SmoothL1Loss()

    def embedding_consistency_loss(self, embeddings, instance_ids):
        """
        Encourages embeddings of the same object to be close.
        embeddings: (B, D, H, W)
        instance_ids: (B, H, W) where 0 is background
        """
        # Note: This is a simplified version. In a real scenario, you'd pull out
        # specific object embeddings based on detected/GT boxes.
        loss = 0
        unique_ids = torch.unique(instance_ids)
        unique_ids = unique_ids[unique_ids != 0] # Exclude background
        
        if len(unique_ids) < 2:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        # Contrastive sample for temporal stability
        # (This would ideally use embeddings from consecutive frames)
        return loss

    def forward(self, preds, targets):
        """
        preds: dict from AccelSightNet
        targets: dict containing gt_objectness, gt_bbox, gt_velocity, gt_ids
        """
        loss_obj = self.obj_loss(preds["objectness"], targets["gt_objectness"])
        
        # Weighted loss only where objects exist
        mask = targets["gt_objectness"] > 0.5
        
        loss_bbox = torch.tensor(0.0, device=preds["bbox"].device)
        loss_vel = torch.tensor(0.0, device=preds["velocity"].device)
        
        if mask.any():
            # Apply mask across channel dimensions
            mask_bbox = mask.expand_as(preds["bbox"])
            loss_bbox = self.bbox_loss(preds["bbox"][mask_bbox], targets["gt_bbox"][mask_bbox])
            
            mask_vel = mask.expand_as(preds["velocity"])
            loss_vel = self.vel_loss(preds["velocity"][mask_vel], targets["gt_velocity"][mask_vel])

        # Total multi-task loss
        total_loss = (self.w_obj * loss_obj + 
                      self.w_bbox * loss_bbox + 
                      self.w_vel * loss_vel)
        
        return {
            "total_loss": total_loss,
            "obj_loss": loss_obj,
            "bbox_loss": loss_bbox,
            "vel_loss": loss_vel
        }
