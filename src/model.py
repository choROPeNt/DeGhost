import torch
import torch.nn as nn
import torch.nn.functional as F



def _safe_groups(c: int, max_groups: int) -> int:
    g = min(max_groups, c)
    while g > 1 and (c % g) != 0:
        g -= 1
    return max(g, 1)


class ResBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int = 3, max_groups: int = 8):
        super().__init__()
        p = k // 2
        g = _safe_groups(c_out, max_groups)

        self.conv1 = nn.Conv2d(c_in, c_out, k, padding=p)
        self.gn1   = nn.GroupNorm(g, c_out)
        self.act1  = nn.SiLU(inplace=True)

        self.conv2 = nn.Conv2d(c_out, c_out, k, padding=p)
        self.gn2   = nn.GroupNorm(g, c_out)
        self.act2  = nn.SiLU(inplace=True)

        self.skip = nn.Identity() if c_in == c_out else nn.Conv2d(c_in, c_out, 1)

        # start near identity
        nn.init.zeros_(self.conv2.weight)
        if self.conv2.bias is not None:
            nn.init.zeros_(self.conv2.bias)

    def forward(self, x):
        h = self.act1(self.gn1(self.conv1(x)))
        h = self.gn2(self.conv2(h))
        return self.act2(h + self.skip(x))


class Downsample(nn.Module):
    def __init__(self, c_in: int, c_out: int, max_groups: int = 8):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1)
        g = _safe_groups(c_out, max_groups)
        self.gn  = nn.GroupNorm(g, c_out)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.gn(self.conv(x)))


class Upsample(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=3, padding=1)

    def forward(self, x, size_hw=None):
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        if size_hw is not None:
            x = F.interpolate(x, size=size_hw, mode="bilinear", align_corners=False)
        return self.conv(x)


class DeGhostUNet(nn.Module):
    """
    Residual U-Net with adjustable levels (downsample steps).

    levels=2 -> bottleneck at H/4
    levels=3 -> bottleneck at H/8 (more context; good for 256 crops)
    """
    def __init__(
        self,
        base: int = 64,
        in_ch: int = 1,
        out_ch: int = 1,
        levels: int = 3,
        max_groups: int = 8,
        mid_blocks: int = 2,
        pad_to_multiple: bool = False,
    ):
        super().__init__()
        assert levels >= 1
        self.levels = levels
        self.pad_to_multiple = pad_to_multiple
        self.in_ch = in_ch
        self.base = base
        # channel sizes per level: [base, 2base, 4base, ...]
        ch = [base * (2 ** i) for i in range(levels)]

        # Encoder: ResBlock at each level + Downsample between levels
        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()

        self.enc_blocks.append(ResBlock(in_ch, ch[0], max_groups=max_groups))
        for i in range(1, levels):
            self.downs.append(Downsample(ch[i - 1], ch[i], max_groups=max_groups))
            self.enc_blocks.append(ResBlock(ch[i], ch[i], max_groups=max_groups))

        # Bottleneck
        mid = []
        for _ in range(mid_blocks):
            mid.append(ResBlock(ch[-1], ch[-1], max_groups=max_groups))
        self.mid = nn.Sequential(*mid)

        # Decoder: Upsample + ResBlock with skip concat
        self.ups = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()

        for i in reversed(range(levels - 1)):
            self.ups.append(Upsample(ch[i + 1], ch[i]))
            self.dec_blocks.append(ResBlock(ch[i] + ch[i], ch[i], max_groups=max_groups))

        self.out = nn.Conv2d(ch[0], out_ch, kernel_size=1)

    def _pad_to(self, x, mult: int):
        B, C, H, W = x.shape
        H2 = ((H + mult - 1) // mult) * mult
        W2 = ((W + mult - 1) // mult) * mult
        pad_h = H2 - H
        pad_w = W2 - W
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x, (H, W)

    def forward(self, ghost: torch.Tensor) -> torch.Tensor:
        orig_hw = None
        if self.pad_to_multiple:
            mult = 2 ** self.levels
            ghost, orig_hw = self._pad_to(ghost, mult)

        skips = []

        # level 0
        x = self.enc_blocks[0](ghost)
        skips.append(x)

        # levels 1..L-1
        for i in range(1, self.levels):
            x = self.downs[i - 1](x)
            x = self.enc_blocks[i](x)
            skips.append(x)

        # bottleneck
        x = self.mid(x)

        # decode (levels-1 up steps)
        for k in range(self.levels - 1):
            skip = skips[-2 - k]  # corresponding encoder skip
            x = self.ups[k](x, size_hw=skip.shape[-2:])
            x = self.dec_blocks[k](torch.cat([x, skip], dim=1))

        residual = self.out(x)

        if orig_hw is not None:
            H, W = orig_hw
            residual = residual[..., :H, :W]

        return residual