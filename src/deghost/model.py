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
        base: int = 32,
        in_ch: int = 1,
        out_ch: int = 1,
        levels: int = 3,
        max_groups: int = 8,
        mid_blocks: int = 2,
        pad_to_multiple: bool = False,
        deep_supervision: bool = True,
    ):
        super().__init__()
        assert levels >= 1
        self.levels = levels
        self.pad_to_multiple = pad_to_multiple
        self.deep_supervision = deep_supervision


        ch = [base * (2 ** i) for i in range(levels)]

        # Encoder
        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()

        self.enc_blocks.append(ResBlock(in_ch, ch[0], max_groups=max_groups))
        for i in range(1, levels):
            self.downs.append(Downsample(ch[i - 1], ch[i], max_groups=max_groups))
            self.enc_blocks.append(ResBlock(ch[i], ch[i], max_groups=max_groups))

        # Bottleneck
        self.mid = nn.Sequential(*[
            ResBlock(ch[-1], ch[-1], max_groups=max_groups) for _ in range(mid_blocks)
        ])

        # Decoder
        self.ups = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()

        for i in reversed(range(levels - 1)):
            self.ups.append(Upsample(ch[i + 1], ch[i]))
            self.dec_blocks.append(ResBlock(ch[i] + ch[i], ch[i], max_groups=max_groups))

        # Main output head (full res)
        self.out = nn.Conv2d(ch[0], out_ch, kernel_size=1)

        # NEW: auxiliary heads (one per decoder stage)
        # dec_blocks[k] outputs ch[i] where i goes: levels-2 ... 0
        # We'll output residuals at those scales too.
        if self.deep_supervision:
            self.aux_out = nn.ModuleList([
                nn.Conv2d(ch_i, out_ch, kernel_size=1)
                for ch_i in [ch[i] for i in reversed(range(levels - 1))]
            ])
        else:
            self.aux_out = None

    def _pad_to(self, x, mult: int):
        B, C, H, W = x.shape
        H2 = ((H + mult - 1) // mult) * mult
        W2 = ((W + mult - 1) // mult) * mult
        pad_h = H2 - H
        pad_w = W2 - W
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x, (H, W)

    def forward(self, ghost: torch.Tensor, return_aux: bool = False):
        """
        Returns:
          residual: (B,out_ch,H,W)
          aux_residuals (optional): list of residuals at intermediate decoder scales,
                                   ordered from coarse->fine or fine->coarse (see below)
        """
        orig_hw = None
        if self.pad_to_multiple:
            mult = 2 ** self.levels
            ghost, orig_hw = self._pad_to(ghost, mult)

        skips = []
        aux = []

        # Encoder
        x = self.enc_blocks[0](ghost)
        skips.append(x)

        for i in range(1, self.levels):
            x = self.downs[i - 1](x)
            x = self.enc_blocks[i](x)
            skips.append(x)

        # Bottleneck
        x = self.mid(x)

        # Decoder
        for k in range(self.levels - 1):
            skip = skips[-2 - k]
            x = self.ups[k](x, size_hw=skip.shape[-2:])
            x = self.dec_blocks[k](torch.cat([x, skip], dim=1))

            if self.deep_supervision and return_aux:
                assert self.aux_out is not None
                aux.append(self.aux_out[k](x))

        residual = self.out(x)

        if orig_hw is not None:
            H, W = orig_hw
            residual = residual[..., :H, :W]
            if self.deep_supervision and return_aux:
                aux = [a[..., :H, :W] if a.shape[-2:] == (H, W) else a for a in aux]

        if self.deep_supervision and return_aux:
            # aux currently goes from coarse->fine? Actually:
            # k=0 is first upsample (deepest)-> next scale (coarsest decoded)
            # k increases => finer. So aux is coarse->fine already.
            return residual, aux

        return residual