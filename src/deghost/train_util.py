from dataclasses import dataclass
from typing import List, Optional

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast

import math

import matplotlib.pyplot as plt
import time

from .loss import ssim_loss, charbonnier, laplacian

@dataclass
class TrainConfig:
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 0.0
    amp: bool = True
    log_every: int = 50
    grad_clip: float = 1.0

    # loss weights
    edge_w: float = 0.05
    ssim_w: float = 0.10
    charb_eps: float = 1e-3
    # NEW: aux / deep supervision
    aux_w: float = 0.0
    aux_weights: Optional[List[float]] = None
    aux_apply_to: str = "residual"  # "residual" or "clean"

    # scheduler
    sched: str = "cosine"          # "cosine" or "plateau" or "none"
    warmup_steps: int = 500        # cosine only
    min_lr: float = 1e-5           # cosine only
    plateau_patience: int = 10     # plateau only
    plateau_factor: float = 0.5    # plateau only


@torch.no_grad()
def evaluate_residual(model, loader, device):
    """
    Returns:
      res_loss   : residual Charbonnier (pred_res vs target_res)
      base_loss  : identity baseline L1 (ghost vs clean)
      edge_loss  : Laplacian L1 on reconstructed clean
      ssim_mean  : mean SSIM(pred_clean, clean)
    """

    model.eval()
    tot_res, tot_base, tot_edge, tot_ssim, n = 0.0, 0.0, 0.0, 0.0, 0

    for ghost, clean, _meta in loader:
        ghost = ghost.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)

        target_res = clean - ghost
        pred_res = model(ghost)

        pred_clean = (ghost + pred_res).clamp(0, 1)

        # --- losses/metrics ---
        res_loss  = charbonnier(pred_res - target_res).mean()
        base_loss = F.l1_loss(ghost, clean)  # identity baseline

        edge_loss = (laplacian(pred_clean) - laplacian(clean)).abs().mean()
        ssim_mean = 1.0 - ssim_loss(pred_clean, clean)  # SSIM in [0,1]

        bs = ghost.size(0)
        tot_res  += float(res_loss) * bs
        tot_base += float(base_loss) * bs
        tot_edge += float(edge_loss) * bs
        tot_ssim += float(ssim_mean) * bs
        n += bs

    return (
        tot_res / max(n, 1),
        tot_base / max(n, 1),
        tot_edge / max(n, 1),
        tot_ssim / max(n, 1),
    )


def train_deghost_residual(model, train_loader, val_loader,test_image, device, cfg: TrainConfig):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    use_amp = bool(cfg.amp and device.type == "cuda")
    scaler = GradScaler(enabled=use_amp)

    EDGE_W = getattr(cfg, "edge_w", 0.05)
    SSIM_W = getattr(cfg, "ssim_w", 0.10)
    EPS    = getattr(cfg, "charb_eps", 1e-3)

    # ---------- Scheduler setup ----------
    steps_per_epoch = len(train_loader)
    total_steps = max(1, cfg.epochs * steps_per_epoch)

    sched_kind = getattr(cfg, "sched", "cosine")

    if sched_kind == "cosine":
        warmup_steps = int(getattr(cfg, "warmup_steps", 0))
        min_lr = float(getattr(cfg, "min_lr", 1e-5))

        def lr_lambda(step):
            # step starts at 0
            if warmup_steps > 0 and step < warmup_steps:
                return (step + 1) / warmup_steps  # linear warmup to 1.0
            # cosine decay from 1.0 -> min_lr/lr
            t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, t))))
            # scale so final lr = min_lr
            return (min_lr / cfg.lr) + (1.0 - (min_lr / cfg.lr)) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
        plateau_scheduler = None
    elif sched_kind == "plateau":
        plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=float(getattr(cfg, "plateau_factor", 0.5)),
            patience=int(getattr(cfg, "plateau_patience", 10)),
            min_lr=float(getattr(cfg, "min_lr", 1e-6)),
        )
        scheduler = None
    else:
        plateau_scheduler = None
        scheduler = None
    # -------------------------------------

    history = {
        "train_total": [],
        "train_res": [],
        "train_edge": [],
        "train_aux": [],
        "train_ssim_loss": [],
        "val_res": [],
        "val_base": [],
        "val_edge": [],
        "val_ssim": [],
        "lr": [],
    }

    best_val = float("inf")
    best_state = None
    global_step = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.time()

        tot_loss = tot_res = tot_edge = tot_ssimL = tot_aux = 0.0
        seen = 0

        # deep supervision weights (coarse->fine). If not provided, auto-decay.
        DS_W = getattr(cfg, "aux_w", 0.0)  # global multiplier, e.g. 1.0
        DS_WEIGHTS = getattr(cfg, "aux_weights", None)  # e.g. [0.5,0.25,0.125]

        for ghost, clean, _meta in train_loader:
            ghost = ghost.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            target_res = clean - ghost

            opt.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", enabled=use_amp):
                if DS_W > 0:
                    pred_res, aux_list = model(ghost,return_aux=True)
                else:
                    pred_res = model(ghost)
                    aux_list = []

                pred_clean = (ghost + pred_res).clamp(0, 1)

                # --- main losses ---
                loss_res  = charbonnier(pred_res - target_res, eps=EPS).mean()
                loss_edge = (laplacian(pred_clean) - laplacian(clean)).abs().mean()
                loss_ssim = ssim_loss(pred_clean, clean)

                # --- aux/deep supervision on residuals ---
                loss_aux = pred_res.new_tensor(0.0)
                if DS_W > 0 and len(aux_list) > 0:
                    if DS_WEIGHTS is None:
                        # auto: 0.5, 0.25, 0.125, ...
                        weights = [0.5 ** (i + 1) for i in range(len(aux_list))]
                    else:
                        weights = list(DS_WEIGHTS)[: len(aux_list)]
                        if len(weights) < len(aux_list):
                            # pad if user provided too few
                            weights += [weights[-1]] * (len(aux_list) - len(weights))

                    for w, aux_res in zip(weights, aux_list):
                        # downsample target residual to aux resolution
                        targ = F.interpolate(target_res, size=aux_res.shape[-2:], mode="area")
                        loss_aux = loss_aux + float(w) * charbonnier(aux_res - targ, eps=EPS).mean()

                    loss_aux = DS_W * loss_aux

                loss = loss_res + EDGE_W * loss_edge + SSIM_W * loss_ssim + loss_aux

            if use_amp:
                scaler.scale(loss).backward()
                if getattr(cfg, "grad_clip", 0.0) and cfg.grad_clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                if getattr(cfg, "grad_clip", 0.0) and cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step()

            # ---- scheduler step (cosine: every step) ----
            global_step += 1
            if scheduler is not None and sched_kind == "cosine":
                scheduler.step()
            # --------------------------------------------

            bs = ghost.size(0)
            seen += bs

            tot_loss  += float(loss.detach()) * bs
            tot_res   += float(loss_res.detach()) * bs
            tot_edge  += float(loss_edge.detach()) * bs
            tot_ssimL += float(loss_ssim.detach()) * bs
            tot_aux   += float(loss_aux.detach()) * bs

            if cfg.log_every and (global_step % cfg.log_every == 0):
                lr_now = opt.param_groups[0]["lr"]
                print(
                    f"epoch {epoch:03d} step {global_step:06d}  "
                    f"lr {lr_now:.2e}  "
                    f"loss {tot_loss/max(seen,1):.5f}  "
                    f"res {tot_res/max(seen,1):.5f}  "
                    f"aux {tot_aux/max(seen,1):.5f}  "
                    f"edge {tot_edge/max(seen,1):.5f}  "
                    f"ssimL {tot_ssimL/max(seen,1):.5f}"
                )

        train_total = tot_loss / max(seen, 1)
        train_res   = tot_res / max(seen, 1)
        train_edge  = tot_edge / max(seen, 1)
        train_aux   = tot_aux / max(seen,1) 
        train_ssimL = tot_ssimL / max(seen, 1)

        history["train_total"].append(train_total)
        history["train_res"].append(train_res)
        history["train_edge"].append(train_edge)
        history["train_aux"].append(train_aux)
        history["train_ssim_loss"].append(train_ssimL)

        # validation metrics
        if val_loader is not None:
            val_res, val_base, val_edge, val_ssim = evaluate_residual(model, val_loader, device)
            history["val_res"].append(val_res)
            history["val_base"].append(val_base)
            history["val_edge"].append(val_edge)
            history["val_ssim"].append(val_ssim)

            # plateau scheduler step (after val)
            if plateau_scheduler is not None and sched_kind == "plateau":
                plateau_scheduler.step(val_res)

            if val_res < best_val:
                best_val = val_res
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            val_res = val_base = val_edge = float("nan")
            val_ssim = float("nan")
        
        if test_image is not None:
            model.eval()

            x = test_image.to(device, non_blocking=True)

            with torch.no_grad():
                pred_res = model(x)                         # (1,1,H,W)
                pred_clean = (x + pred_res).clamp(0, 1)

            # to numpy (H,W)
            x_np = x[0, 0].detach().cpu().numpy()
            res_np = pred_res[0, 0].detach().cpu().numpy()
            clean_np = pred_clean[0, 0].detach().cpu().numpy()

            fig, axs = plt.subplots(1, 3, figsize=(20, 4))

            im0 = axs[0].imshow(x_np, cmap="gray", vmin=0, vmax=1)
            axs[0].set_title("input")
            fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

            im1 = axs[1].imshow(res_np, cmap="plasma")
            axs[1].set_title("pred_res")
            fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

            im2 = axs[2].imshow(clean_np, cmap="gray", vmin=0, vmax=1)
            axs[2].set_title("pred_clean")
            fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

            for ax in axs:
                ax.axis("off")

            plt.tight_layout()

            # save as PNG with epoch in filename
            out_path = Path("out")
            out_path.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path / f"test_{epoch:03d}.png", dpi=200, bbox_inches="tight")
            plt.close(fig)   # avoid figure buildup


        lr_now = opt.param_groups[0]["lr"]
        history["lr"].append(lr_now)

        dt = time.time() - t0
        print(
            f"[Epoch {epoch:03d}]  "
            f"LR: {lr_now:>8.2e}  |  "
            f"Train → total: {train_total:>8.5f}  "
            f"(res: {train_res:>7.5f}  "
            f"edge: {train_edge:>7.5f}  "
            f"aux: {train_aux:>7.5f}  "
            f"ssimL: {train_ssimL:>7.5f})  |  "
            f"Val → res: {val_res:>7.5f}  "
            f"base: {val_base:>7.5f}  "
            f"edge: {val_edge:>7.5f}  "
            f"ssim: {val_ssim:>6.4f}  |  "
            f"time: {dt:>5.1f}s"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    return history
