from einops import repeat
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalDiceLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, dice_weight=0.5):
        """
        结合 Focal Loss 和 Dice Loss
        Args:
            alpha (float): Focal Loss 的平衡因子，默认 0.25
            gamma (float): Focal Loss 的调节因子，默认 2.0
            dice_weight (float): Dice Loss 在总 Loss 中的权重（范围 0~1）
        """
        super(FocalDiceLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.dice_weight = dice_weight

    def forward(self, pred_mask, target_mask):
        """
        Args:
            pred_mask (Tensor): 预测的 mask，形状 (bs, 1, h, w)
            target_mask (Tensor): 真实 mask，形状 (bs, 1, h, w)
        Returns:
            loss (Tensor): 组合后的 Loss
        """
        assert len(pred_mask.shape) == 4

        # 1. 计算 Dice Loss
        pred_mask = torch.sigmoid(pred_mask)  # 确保输出值在 (0,1) 之间
        pred_flat = pred_mask.flatten(1)
        target_flat = target_mask.flatten(1)
        intersection = (pred_flat * target_flat).sum(-1)
        dice = 1 - (2 * intersection + 1e-6) / (pred_flat.sum(-1) + target_flat.sum(-1) + 1e-6)
        dice_loss = dice.mean()

        # 2. 计算 Focal Loss
        bce_loss = F.binary_cross_entropy(pred_mask, target_mask, reduction='none')
        focal_weight = self.alpha * (1 - pred_mask) ** self.gamma * target_mask + \
                       (1 - self.alpha) * pred_mask ** self.gamma * (1 - target_mask)
        focal_loss = (focal_weight * bce_loss).mean()

        # 3. 组合 Loss
        loss = self.dice_weight * dice_loss + (1 - self.dice_weight) * focal_loss
        return loss

def F5_IoU_BCELoss(pred_mask, five_gt_masks):
    """
    binary cross entropy loss (iou loss) of the total five frames for multiple sound source segmentation

    Args:
    pred_mask: predicted masks for a batch of data, shape:[bs*5, 1, 224, 224]
    five_gt_masks: ground truth mask of the total five frames, shape: [bs*5, 1, 224, 224]
    """
    assert len(pred_mask.shape) == 4
    pred_mask = torch.sigmoid(pred_mask)  # [bs*5, 1, 224, 224]
    # five_gt_masks = five_gt_masks.view(-1, 1, five_gt_masks.shape[-2], five_gt_masks.shape[-1]) # [bs*5, 1, 224, 224]
    loss = nn.BCELoss()(pred_mask, five_gt_masks)

    return loss


def F5_Dice_loss(pred_mask, five_gt_masks):
    """dice loss for aux loss

    Args:
        pred_mask (Tensor): (bs, 1, h, w)
        five_gt_masks (Tensor): (bs, 1, h, w)
    """
    assert len(pred_mask.shape) == 4
    pred_mask = torch.sigmoid(pred_mask)

    pred_mask = pred_mask.flatten(1)
    gt_mask = five_gt_masks.flatten(1)
    a = (pred_mask * gt_mask).sum(-1)
    b = (pred_mask * pred_mask).sum(-1) + 0.001
    c = (gt_mask * gt_mask).sum(-1) + 0.001
    d = (2 * a) / (b + c)
    loss = 1 - d
    return loss.mean()


def ssim_loss_func(pred, target, C1=0.01**2, C2=0.03**2):
    """
    Compute SSIM loss for tensors of shape [B, N, H, W].
    Args:
        pred: Tensor of shape [B, N, H, W] (predicted masks)
        target: Tensor of shape [B, N, H, W] (ground truth masks)
        C1: Stability constant for mean terms
        C2: Stability constant for variance terms
    Returns:
        SSIM loss: A scalar value representing (1 - SSIM)
    """
    B, N, H, W = pred.shape

    # Mean calculation
    mu_x = F.avg_pool2d(pred, kernel_size=3, stride=1, padding=1)
    mu_y = F.avg_pool2d(target, kernel_size=3, stride=1, padding=1)

    # Variance calculation
    sigma_x = F.avg_pool2d(pred**2, kernel_size=3, stride=1, padding=1) - mu_x**2
    sigma_y = F.avg_pool2d(target**2, kernel_size=3, stride=1, padding=1) - mu_y**2
    sigma_xy = F.avg_pool2d(pred * target, kernel_size=3, stride=1, padding=1) - mu_x * mu_y

    # SSIM calculation
    ssim_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    ssim_d = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)
    ssim_map = ssim_n / (ssim_d + 1e-6)

    # Compute mean SSIM over all channels and batches
    ssim_value = ssim_map.mean(dim=[1, 2, 3])  # Per channel SSIM
    return 1 - ssim_value.mean()  # Return (1 - SSIM)


import torch
import torch.nn.functional as F

def dep_mask_supervised_loss_focaldice(dep_mask_pred, dep_mask_gt, loss_weight_bg=0.2):
    """
    dep_mask_pred: (B, 1, H, W) 预测的dep mask（logits）
    dep_mask_gt:   (B, 1, H, W) 依赖关系的GT（0或1）
    loss_weight_bg: 背景loss权重
    alpha, gamma, smooth: FocalDice参数
    """
    focal_dice_loss = FocalDiceLoss(alpha=0.25, gamma=2.0, dice_weight=0.4)
    fg_mask = (dep_mask_gt == 1)
    bg_mask = (dep_mask_gt == 0)

    # 前景：FocalDiceLoss
    if fg_mask.sum() > 0:
        fg_loss = focal_dice_loss(dep_mask_pred, dep_mask_gt)
    else:
        fg_loss = 0.0

    # 背景：BCE
    if bg_mask.sum() > 0:
        bg_loss = loss_weight_bg * F.binary_cross_entropy_with_logits(dep_mask_pred[bg_mask], torch.zeros_like(dep_mask_pred[bg_mask]))
    else:
        bg_loss = 0.0

    return fg_loss + bg_loss

def IouSemanticAwareLoss(mask_func,mask_dep,func_gt,dep_gt,func_feat, dep_feat,imgs, weight_dict, loss_type='bce', **kwargs):
    total_loss = 0
    loss_dict = {}
    """
    if loss_type == 'bce':
        loss_func = F5_IoU_BCELoss
    elif loss_type == 'dice':
        loss_func = F5_Dice_loss
    else:
        raise ValueError
    """
    loss_func_func = FocalDiceLoss(alpha=0.25, gamma=2.0, dice_weight=0.4)

    iou_loss_func = weight_dict['iou_loss_func'] * loss_func_func(mask_func, func_gt)
    iou_loss_dep = weight_dict['iou_loss_dep'] * dep_mask_supervised_loss_focaldice(mask_dep, dep_gt)

    total_loss += iou_loss_func
    total_loss += iou_loss_dep
    loss_dict['iou_loss_func'] = iou_loss_func.item()
    loss_dict['iou_loss_dep'] = iou_loss_dep.item()


    func_feat = torch.mean(func_feat, dim=1, keepdim=True)
    func_feat = F.interpolate(
        func_feat, func_gt.shape[-2:], mode='bilinear', align_corners=False)
    mix_loss_func = weight_dict['mix_loss_func']*loss_func_func(func_feat, func_gt)

    dep_feat = torch.mean(dep_feat, dim=1, keepdim=True)
    dep_feat = F.interpolate(
        dep_feat, func_gt.shape[-2:], mode='bilinear', align_corners=False)
    mix_loss_dep = weight_dict['mix_loss_dep']*dep_mask_supervised_loss_focaldice(dep_feat, dep_gt)

    total_loss += mix_loss_func
    total_loss += mix_loss_dep

    loss_dict['mix_loss_func'] = mix_loss_func.item()
    loss_dict['mix_loss_dep'] = mix_loss_dep.item()

    return total_loss, loss_dict
