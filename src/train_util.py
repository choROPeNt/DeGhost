
import torch
import torch.nn.functional as F
from .loss import ssim_loss, charbonnier, laplacian


from dataclasses import dataclass

import time

@dataclass
class TrainConfig:
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    amp: bool = True              # CUDA only
    grad_clip: float = 1.0        # 0 disables
    edge_w: float = 0.05
    ssim_w: float = 0.10
    charb_eps: float = 1e-3


@torch.no_grad()
def evaluate_residual(model, loader, device, edge_w: float = 0.05, eps: float = 1e-3):
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




def train_deghost_residual(model, train_loader, val_loader, device, cfg: TrainConfig):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    use_amp = bool(cfg.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    EDGE_W = getattr(cfg, "edge_w", 0.05)   # add cfg.edge_w if you like
    SSIM_W = getattr(cfg, "ssim_w", 0.10)   # add cfg.ssim_w if you like
    EPS    = getattr(cfg, "charb_eps", 1e-3)

    history = {
        "train_total": [],
        "train_res": [],
        "train_edge": [],
        "train_ssim_loss": [],
        "val_res": [],
        "val_base": [],
        "val_edge": [],
        "val_ssim": [],
    }

    best_val = float("inf")
    best_state = None
    global_step = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.time()

        tot_loss = tot_res = tot_edge = tot_ssimL = 0.0
        seen = 0

        for ghost, clean, _meta in train_loader:
            ghost = ghost.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            target_res = clean - ghost

            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                pred_res = model(ghost)
                pred_clean = (ghost + pred_res).clamp(0, 1)

                # --- residual loss (Charbonnier) ---
                loss_res = charbonnier(pred_res - target_res, eps=EPS).mean()

                # --- edge loss (Laplacian on clean recon) ---
                loss_edge = (laplacian(pred_clean) - laplacian(clean)).abs().mean()

                # --- SSIM loss (1-SSIM) on clean recon ---
                loss_ssim = ssim_loss(pred_clean, clean)

                loss = loss_res + EDGE_W * loss_edge + SSIM_W * loss_ssim

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

            bs = ghost.size(0)
            seen += bs
            global_step += 1

            tot_loss  += float(loss.detach()) * bs
            tot_res   += float(loss_res.detach()) * bs
            tot_edge  += float(loss_edge.detach()) * bs
            tot_ssimL += float(loss_ssim.detach()) * bs

            # if cfg.log_every and (global_step % cfg.log_every == 0):
            #     print(
            #         f"epoch {epoch:03d} step {global_step:06d}  "
            #         f"loss {tot_loss/max(seen,1):.5f}  "
            #         f"res {tot_res/max(seen,1):.5f}  "
            #         f"edge {tot_edge/max(seen,1):.5f}  "
            #         f"ssimL {tot_ssimL/max(seen,1):.5f}"
            #     )

        # epoch train metrics
        train_total = tot_loss / max(seen, 1)
        train_res   = tot_res / max(seen, 1)
        train_edge  = tot_edge / max(seen, 1)
        train_ssimL = tot_ssimL / max(seen, 1)

        history["train_total"].append(train_total)
        history["train_res"].append(train_res)
        history["train_edge"].append(train_edge)
        history["train_ssim_loss"].append(train_ssimL)

        # validation metrics (res, base, edge, ssim)
        if val_loader is not None:
            val_res, val_base, val_edge, val_ssim = evaluate_residual(model, val_loader, device, edge_w=EDGE_W)
            history["val_res"].append(val_res)
            history["val_base"].append(val_base)
            history["val_edge"].append(val_edge)
            history["val_ssim"].append(val_ssim)

            # pick "best" by residual loss (or switch to composite if you prefer)
            if val_res < best_val:
                best_val = val_res
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            val_res = val_base = val_edge = float("nan")
            val_ssim = float("nan")

        dt = time.time() - t0
        print(
            f"[epoch {epoch:03d}] "
            f"train total {train_total:.5f} (res {train_res:.5f} + {EDGE_W}*edge {train_edge:.5f} + {SSIM_W}*ssimL {train_ssimL:.5f})  "
            f"val res {val_res:.5f}  base {val_base:.5f}  edge {val_edge:.5f}  ssim {val_ssim:.4f}  "
            f"({dt:.1f}s)"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    return history
