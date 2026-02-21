import torch
import torch.nn.functional as F



def charbonnier(x, eps=1e-3):
    return torch.sqrt(x * x + eps * eps)

def laplacian(x):
    k = torch.tensor(
        [[0, 1, 0],
         [1,-4, 1],
         [0, 1, 0]],
        dtype=x.dtype,
        device=x.device
    )[None, None]
    return F.conv2d(x, k, padding=1)


def ssim_loss(x: torch.Tensor, y: torch.Tensor, window_size: int = 11, C1: float = 0.01**2, C2: float = 0.03**2):
    """
    Returns: 1 - SSIM (so lower is better)
    x,y: (B,1,H,W) in [0,1]
    Uses a uniform window via avg_pool2d (fast and stable).
    """
    pad = window_size // 2

    mu_x = F.avg_pool2d(x, window_size, stride=1, padding=pad)
    mu_y = F.avg_pool2d(y, window_size, stride=1, padding=pad)

    sigma_x  = F.avg_pool2d(x * x, window_size, stride=1, padding=pad) - mu_x * mu_x
    sigma_y  = F.avg_pool2d(y * y, window_size, stride=1, padding=pad) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x * y, window_size, stride=1, padding=pad) - mu_x * mu_y

    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / ((mu_x * mu_x + mu_y * mu_y + C1) * (sigma_x + sigma_y + C2))
    return 1.0 - ssim_map.mean()