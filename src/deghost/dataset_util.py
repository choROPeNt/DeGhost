from pathlib import Path
from typing import Optional, Tuple, Dict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F



# ----------------------------
# Ghosting augmentation
# ----------------------------
def add_ghost_shadow_safe(
    clean: torch.Tensor,
    alpha: Optional[float] = None,
    dy: Optional[int] = None,
    dx: Optional[int] = None,
    max_shift: int = 1,
    allow_diagonal: bool = True,
    blur_sigma: float = 0.0,
    noise_std: float = 0.0,
    crop_safe: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    clean: (H,W) or (1,H,W) float in [0,1]
    returns: ghost, clean, meta (both possibly center-cropped by max_shift)
    """
    if rng is None:
        rng = np.random.default_rng()

    if clean.ndim == 2:
        clean = clean.unsqueeze(0)

    # choose random shift if not given
    if dy is None or dx is None:
        base = [(1,0),(-1,0),(0,1),(0,-1)]
        if allow_diagonal:
            base += [(1,1),(1,-1),(-1,1),(-1,-1)]
        dy0, dx0 = base[int(rng.integers(0, len(base)))]
        s = 1 if max_shift <= 1 else int(rng.integers(1, max_shift + 1))
        dy, dx = dy0 * s, dx0 * s

    if alpha is None:
        alpha = float(rng.uniform(0.25, 0.75))

    shifted = torch.roll(clean, shifts=(int(dy), int(dx)), dims=(-2, -1))

    # optional blur on shifted copy only (torch-only approx)
    if blur_sigma > 0:
        # simple separable-ish blur via avgpool as a cheap proxy (fast for testing)
        # replace with torchvision.gaussian_blur if you want exact gaussian
        k = max(3, int(round(blur_sigma * 6)) | 1)
        pad = k // 2
        shifted = F.avg_pool2d(shifted.unsqueeze(0), kernel_size=k, stride=1, padding=pad).squeeze(0)

    ghost = (1.0 - float(alpha)) * clean + float(alpha) * shifted

    if noise_std > 0:
        ghost = ghost + torch.randn_like(ghost) * float(noise_std)

    ghost = ghost.clamp(0, 1)

    # safe crop to remove wrap-around from roll
    if crop_safe and max_shift > 0:
        _, H, W = ghost.shape
        m = int(max_shift)
        if H <= 2*m or W <= 2*m:
            raise ValueError(f"crop_safe too large: max_shift={m} for patch {H}x{W}")
        ghost = ghost[:, m:H-m, m:W-m]
        clean = clean[:, m:H-m, m:W-m]

    meta = {
        "alpha": float(alpha),
        "dy": int(dy),
        "dx": int(dx),
        "cropped_by": int(max_shift if crop_safe else 0),
    }
    return ghost, clean, meta


def random_crop_pair(img1: torch.Tensor, img2: torch.Tensor, crop_size: int, rng: np.random.Generator):
    """
    Random crop same region from img1 and img2.
    Both must be (1,H,W).
    """
    _, H, W = img1.shape
    if H < crop_size or W < crop_size:
        raise ValueError(f"Crop size {crop_size} larger than image {H}x{W}.")

    top  = int(rng.integers(0, H - crop_size + 1))
    left = int(rng.integers(0, W - crop_size + 1))

    img1 = img1[:, top:top+crop_size, left:left+crop_size]
    img2 = img2[:, top:top+crop_size, left:left+crop_size]
    return img1, img2


def downscale_by2_pair(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    a,b: (1,H,W) float
    returns (1,H/2,W/2) using area resampling (best for microscopy)
    """
    a4 = a.unsqueeze(0)  # (1,1,H,W)
    b4 = b.unsqueeze(0)
    a_ds = F.interpolate(a4, scale_factor=0.5, mode="area").squeeze(0)
    b_ds = F.interpolate(b4, scale_factor=0.5, mode="area").squeeze(0)
    return a_ds, b_ds


# ----------------------------
# Dataset
# ----------------------------
class PatchNPYDataset(Dataset):
    """
    Loads pre-extracted patches from a .npy/.npz and applies ghosting transform.
    Returns:
      ghost: (1,h,w) float32 in [0,1]
      clean: (1,h,w) float32 in [0,1]
      meta: dict of tensors (alpha,dx,dy,cropped_by)
    """
    def __init__(
        self,
        patches_path: str | Path,
        normalize: str = "uint8",
        max_shift: int = 8,
        allow_diagonal: bool = True,
        alpha_range: Tuple[float, float] = (0.6, 0.9),
        blur_sigma_range: Tuple[float, float] = (0.0, 0.0),
        noise_std_range: Tuple[float, float] = (0.0, 0.0),
        crop_safe: bool = True,
        random_flip_rot: bool = True,
        crop_size: Optional[int] = 256,
        downscale_factor2: bool = True,     # <--- requested
        seed: int = 0,
    ):
        self.patches_path = Path(patches_path)
        if not self.patches_path.exists():
            raise FileNotFoundError(self.patches_path)

        arr = np.load(self.patches_path, allow_pickle=False)
        if isinstance(arr, np.lib.npyio.NpzFile):
            key = list(arr.keys())[0]
            arr = arr[key]

        if arr.ndim == 4 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        if arr.ndim != 3:
            raise ValueError(f"Expected (N,H,W) or (N,H,W,1), got {arr.shape}")

        self.patches = arr
        self.normalize = normalize

        self.max_shift = int(max_shift)
        self.allow_diagonal = allow_diagonal
        self.alpha_range = alpha_range
        self.blur_sigma_range = blur_sigma_range
        self.noise_std_range = noise_std_range
        self.crop_safe = crop_safe

        self.random_flip_rot = random_flip_rot
        self.crop_size = crop_size
        self.downscale_factor2 = downscale_factor2

        self.seed = int(seed)

        # shift directions
        self.base_dirs = [(1,0),(-1,0),(0,1),(0,-1)]
        if self.allow_diagonal:
            self.base_dirs += [(1,1),(1,-1),(-1,1),(-1,-1)]

    def __len__(self) -> int:
        return self.patches.shape[0]

    def _to_tensor01(self, x_np: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(x_np)
        if x.ndim != 2:
            raise ValueError("Patch must be 2D after loading.")
        x = x.to(torch.float32)

        if self.normalize == "uint8":
            x = x / 255.0
        elif self.normalize == "none":
            pass
        else:
            raise ValueError(f"Unknown normalize='{self.normalize}'")

        return x.clamp(0, 1)

    def _augment_geom(self, x: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
        # x: (H,W)
        if not self.random_flip_rot:
            return x
        k = int(rng.integers(0, 4))
        x = torch.rot90(x, k, dims=(0, 1))
        if rng.random() < 0.5:
            x = torch.flip(x, dims=(1,))
        if rng.random() < 0.5:
            x = torch.flip(x, dims=(0,))
        return x

    def __getitem__(self, idx: int):
        # per-sample RNG (safe with multi-workers)
        rng = np.random.default_rng(self.seed + idx)

        clean = self._to_tensor01(self.patches[idx])
        clean = self._augment_geom(clean, rng)          # (H,W)
        clean = clean.unsqueeze(0)                      # (1,H,W)



        # sample ghost params
        alpha = float(rng.uniform(*self.alpha_range))

        dy0, dx0 = self.base_dirs[int(rng.integers(0, len(self.base_dirs)))]
        s = 1 if self.max_shift <= 1 else int(rng.integers(1, self.max_shift + 1))
        dy, dx = int(dy0 * s), int(dx0 * s)

        blur_sigma = float(rng.uniform(*self.blur_sigma_range))
        noise_std  = float(rng.uniform(*self.noise_std_range))

        ghost, clean_out, meta = add_ghost_shadow_safe(
            clean,
            alpha=alpha,
            dy=dy, dx=dx,
            max_shift=self.max_shift,
            allow_diagonal=self.allow_diagonal,
            blur_sigma=blur_sigma,
            noise_std=noise_std,
            crop_safe=self.crop_safe,
            rng=rng,
        )

        # random crop (after safe-crop)
        if self.crop_size is not None:
            ghost, clean_out = random_crop_pair(ghost, clean_out, crop_size=self.crop_size, rng=rng)
        # downscale AFTER cropping (for test training)
        if self.downscale_factor2:
            ghost, clean_out = downscale_by2_pair(ghost, clean_out)

            # also scale shift meta to the new pixel grid (optional, but correct)
            meta["dx"] = int(round(meta["dx"] / 2))
            meta["dy"] = int(round(meta["dy"] / 2))
            meta["cropped_by"] = int(round(meta["cropped_by"] / 2))

        # meta -> tensors so default collate works nicely
        meta_t = {
            "alpha": torch.tensor(meta["alpha"], dtype=torch.float32),
            "dy": torch.tensor(meta["dy"], dtype=torch.float32),
            "dx": torch.tensor(meta["dx"], dtype=torch.float32),
            "cropped_by": torch.tensor(meta["cropped_by"], dtype=torch.int64),
        }

        return ghost.to(torch.float32), clean_out.to(torch.float32), meta_t


# ----------------------------
# Dataloader helper
# ----------------------------
def make_dataloader(
    data: str | Path,
    batch_size: int = 8,
    num_workers: int = 0,
    shuffle: bool = True,
    pin_memory: bool = False,
    persistent_workers: bool = True,
    **dataset_kwargs,
) -> DataLoader:
    ds = PatchNPYDataset(data, **dataset_kwargs)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(persistent_workers and num_workers > 0),
        drop_last=True,
    )