

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SSIMLoss(nn.Module):
    
    
    def __init__(self, window_size=11, sigma=1.5, channels=1, reduction='mean'):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.channels = channels
        self.reduction = reduction
        
       
        self.register_buffer('window', self._create_window(window_size, sigma, channels))
    
    def _create_window(self, window_size, sigma, channels):
        
        gauss = torch.Tensor([
            np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
            for x in range(window_size)
        ])
        gauss = gauss / gauss.sum()
        
        
        _1D_window = gauss.unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        
        
        window = _2D_window.expand(channels, 1, window_size, window_size).contiguous()
        
        return window
    
    def _ssim(self, pred, target, window, window_size, channels):
        
        mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=channels)
        mu2 = F.conv2d(target, window, padding=window_size // 2, groups=channels)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=channels) - mu1_sq
        sigma2_sq = F.conv2d(target * target, window, padding=window_size // 2, groups=channels) - mu2_sq
        sigma12 = F.conv2d(pred * target, window, padding=window_size // 2, groups=channels) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return ssim_map
    
    def forward(self, pred, target):
        
        channels = pred.size(1)
        
        if channels != self.channels:
            window = self._create_window(self.window_size, self.sigma, channels).to(pred.device)
        else:
            window = self.window.to(pred.device)
        
        ssim_map = self._ssim(pred, target, window, self.window_size, channels)
        
        if self.reduction == 'mean':
            return 1 - ssim_map.mean()
        elif self.reduction == 'sum':
            return (1 - ssim_map).sum()
        else:
            return 1 - ssim_map


class CountLoss(nn.Module):
    
    
    def __init__(self, loss_type='mae', reduction='mean'):
        super().__init__()
        self.loss_type = loss_type
        self.reduction = reduction
    
    def forward(self, pred, target):
        
        pred_count = pred.sum(dim=[1, 2, 3]) 
        target_count = target.sum(dim=[1, 2, 3]) 
        
        if self.loss_type == 'mae':
            loss = torch.abs(pred_count - target_count)
        elif self.loss_type == 'mse':
            loss = (pred_count - target_count) ** 2
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class MSELoss(nn.Module):
    
    
    def __init__(self, reduction='sum'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, pred, target):
        mse = (pred - target) ** 2
        
        if self.reduction == 'sum':
            return mse.sum() / pred.size(0)  
        elif self.reduction == 'mean':
            return mse.mean()
        else:
            return mse


class TotalVariationLoss(nn.Module):
    
    
    def __init__(self):
        super().__init__()
    
    def forward(self, pred):
        """Compute TV loss."""
        diff_h = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        diff_w = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        
        return (diff_h.sum() + diff_w.sum()) / pred.numel()


class CombinedLoss(nn.Module):
    
    def __init__(
        self,
        mse_weight=1.0,
        ssim_weight=0.01,
        count_weight=0.001,
        tv_weight=0.0,
        use_tv=False,
    ):
        
        super().__init__()
        
        self.mse_weight = mse_weight
        self.ssim_weight = ssim_weight
        self.count_weight = count_weight
        self.tv_weight = tv_weight
        self.use_tv = use_tv
        
        self.mse_loss = MSELoss(reduction='sum')
        self.ssim_loss = SSIMLoss()
        self.count_loss = CountLoss(loss_type='mae')
        
        if use_tv:
            self.tv_loss = TotalVariationLoss()
    
    def forward(self, pred, target):
        
        mse = self.mse_loss(pred, target)
        
      
        ssim = self.ssim_loss(pred, target)
        
       
        count = self.count_loss(pred, target)
        
       
        total_loss = (
            self.mse_weight * mse +
            self.ssim_weight * ssim +
            self.count_weight * count
        )
        
       
        if self.use_tv:
            tv = self.tv_loss(pred)
            total_loss += self.tv_weight * tv
        else:
            tv = torch.tensor(0.0)
        
        return total_loss, {
            'mse': mse.item(),
            'ssim': ssim.item(),
            'count': count.item(),
            'tv': tv.item() if isinstance(tv, torch.Tensor) else tv,
        }


class BayesianLoss(nn.Module):
    
    
    def __init__(self, sigma=4.0):
        super().__init__()
        self.sigma = sigma
    
    def forward(self, pred, target):
        
        pred_sum = pred.sum(dim=[1, 2, 3])
        target_sum = target.sum(dim=[1, 2, 3])
        
        
        count_loss = torch.abs(pred_sum - target_sum).mean()
        
        
        density_loss = F.mse_loss(pred, target, reduction='sum') / pred.size(0)
        
        return count_loss + density_loss


class AdaptiveLoss(nn.Module):
    
    
    def __init__(self, base_mse_weight=1.0, base_count_weight=0.001):
        super().__init__()
        self.base_mse_weight = base_mse_weight
        self.base_count_weight = base_count_weight
    
    def forward(self, pred, target):
        batch_size = pred.size(0)
        total_loss = 0
        
        for i in range(batch_size):
            pred_i = pred[i]
            target_i = target[i]
            
        
            gt_count = target_i.sum()
            
        
            if gt_count < 50:
        
                mse_weight = 0.5
                count_weight = 0.01
            elif gt_count < 200:
            
                mse_weight = 1.0
                count_weight = 0.001
            else:
            
                mse_weight = 1.0
                count_weight = 0.0001
            
        
            mse = ((pred_i - target_i) ** 2).sum()
            
    
            count = torch.abs(pred_i.sum() - gt_count)
            
            total_loss += mse_weight * mse + count_weight * count
        
        return total_loss / batch_size


class OTLoss(nn.Module):
    
    
    def __init__(self, lambda_ot=0.1, lambda_tv=0.01):
        super().__init__()
        self.lambda_ot = lambda_ot
        self.lambda_tv = lambda_tv
    
    def forward(self, pred, target):
        
        batch_size = pred.size(0)
        
        
        pred_count = pred.sum(dim=[1, 2, 3])
        target_count = target.sum(dim=[1, 2, 3])
        count_loss = torch.abs(pred_count - target_count).mean()
       
        pred_norm = pred / (pred.sum(dim=[2, 3], keepdim=True) + 1e-8)
        target_norm = target / (target.sum(dim=[2, 3], keepdim=True) + 1e-8)
        
        ot_loss = torch.abs(pred_norm - target_norm).sum(dim=[1, 2, 3]).mean()
        
        diff_h = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :]).sum()
        diff_w = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1]).sum()
        tv_loss = (diff_h + diff_w) / pred.numel()
        
        return count_loss + self.lambda_ot * ot_loss + self.lambda_tv * tv_loss


def create_loss_function(loss_type='combined', **kwargs):
    
    if loss_type == 'mse':
        return MSELoss(reduction='sum')
    elif loss_type == 'combined':
        return CombinedLoss(**kwargs)
    elif loss_type == 'bayesian':
        return BayesianLoss(**kwargs)
    elif loss_type == 'adaptive':
        return AdaptiveLoss(**kwargs)
    elif loss_type == 'ot':
        return OTLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
   
    print("Testing loss functions...")
    
    pred = torch.rand(2, 1, 32, 32) * 10
    target = torch.rand(2, 1, 32, 32) * 10
    
    print("\n1. MSE Loss:")
    loss_fn = MSELoss()
    loss = loss_fn(pred, target)
    print(f"   Loss: {loss.item():.4f}")
    
    print("\n2. SSIM Loss:")
    loss_fn = SSIMLoss()
    loss = loss_fn(pred, target)
    print(f"   Loss: {loss.item():.4f}")
    
    print("\n3. Count Loss:")
    loss_fn = CountLoss()
    loss = loss_fn(pred, target)
    print(f"   Loss: {loss.item():.4f}")
    
    print("\n4. Combined Loss:")
    loss_fn = CombinedLoss()
    loss, components = loss_fn(pred, target)
    print(f"   Total: {loss.item():.4f}")
    print(f"   Components: {components}")
    
    print("\n5. Bayesian Loss:")
    loss_fn = BayesianLoss()
    loss = loss_fn(pred, target)
    print(f"   Loss: {loss.item():.4f}")
    
    print("\nLoss function tests passed!")