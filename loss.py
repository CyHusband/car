import torch
import torch.nn as nn
import torch.nn.functional as F
class MyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pre, target):
        cls_pre = pre[:, 0]
        cls_tar = target[:, 0]
        cls_loss = F.binary_cross_entropy(F.sigmoid(cls_pre), cls_tar)

        box_pre = pre[:, 1:]
        box_tar = target[:, 1:]
        box_loss = self.giou_loss(box_pre, box_tar)
        box_loss = (1 - box_loss) * cls_tar

        loss = (cls_loss + box_loss.mean()) / 2
        return loss

    def giou_loss(self, box1, box2, eps=1e-12):
    
        #xywh
        b1_x1, b1_x2 = box1[:,0] - box1[:,2] / 2, box1[:,0] + box1[:,2] / 2 + eps
        b1_y1, b1_y2 = box1[:,1] - box1[:,3] / 2, box1[:,1] + box1[:,3] / 2 + eps
        b2_x1, b2_x2 = box2[:,0] - box2[:,2] / 2, box2[:,0] + box2[:,2] / 2 + eps
        b2_y1, b2_y2 = box2[:,1] - box2[:,3] / 2, box2[:,1] + box2[:,3] / 2 + eps

        # Intersection area
        inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

        # Union Area
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
        union = w1 * h1 + w2 * h2 - inter

        iou = inter / union  # iou

        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  
        c_area = cw * ch  
        return iou - (c_area - union) / c_area  #giou
