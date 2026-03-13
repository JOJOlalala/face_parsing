import torch
import torch.nn as nn
import torch.nn.functional as F


class DSConv(nn.Module):
    """
    Depthwise-separable conv: depthwise 3x3 + pointwise 1x1.
    Much fewer params than standard conv.
    """

    def __init__(self, in_ch: int, out_ch: int, k: int = 3, p: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(
            in_ch, in_ch, kernel_size=k, padding=p, groups=in_ch, bias=False
        )
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return self.act(x)


class ConvBlock(nn.Module):
    """Two DSConv blocks."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.c1 = DSConv(in_ch, out_ch)
        self.c2 = DSConv(out_ch, out_ch)

    def forward(self, x):
        return self.c2(self.c1(x))


class Down(nn.Module):
    """Downsample + conv block."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.block = ConvBlock(in_ch, out_ch)

    def forward(self, x):
        x = self.pool(x)
        return self.block(x)


class Up(nn.Module):
    """Upsample + concat skip + conv block."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        # After concat: in_ch + skip_ch
        self.block = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        # If shapes mismatch by 1 due to odd sizes, center-crop skip
        if x.shape[-2:] != skip.shape[-2:]:
            dh = skip.size(-2) - x.size(-2)
            dw = skip.size(-1) - x.size(-1)
            skip = skip[
                :,
                :,
                dh // 2 : skip.size(-2) - (dh - dh // 2),
                dw // 2 : skip.size(-1) - (dw - dw // 2),
            ]
        x = torch.cat([skip, x], dim=1)
        return self.block(x)


class LiteUNetDS(nn.Module):
    """
    Lite U-Net for face parsing.
    Good default widths under param budget:
      base=32 -> strong, still typically under 1.82M with DSConv.
    """

    def __init__(self, num_classes: int = 19, base: int = 32):
        super().__init__()
        c1, c2, c3, c4, c5 = base, base * 2, base * 4, base * 8, base * 8

        self.in_block = ConvBlock(3, c1)
        self.down1 = Down(c1, c2)
        self.down2 = Down(c2, c3)
        self.down3 = Down(c3, c4)

        self.bottleneck = Down(c4, c5)

        self.up3 = Up(c5, c4, c3)
        self.up2 = Up(c3, c3, c2)
        self.up1 = Up(c2, c2, c1)
        self.up0 = Up(c1, c1, c1)

        self.head = nn.Conv2d(c1, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.in_block(x)  # (B, c1, H, W)
        x2 = self.down1(x1)  # (B, c2, H/2, W/2)
        x3 = self.down2(x2)  # (B, c3, H/4, W/4)
        x4 = self.down3(x3)  # (B, c4, H/8, W/8)
        xb = self.bottleneck(x4)  # (B, c5, H/16, W/16)

        x = self.up3(xb, x4)  # -> c3
        x = self.up2(x, x3)  # -> c2
        x = self.up1(x, x2)  # -> c1
        x = self.up0(x, x1)  # -> c1

        return self.head(x)  # (B, num_classes, H, W)


# ---------------------------------------------------------------------------
# Standard U-Net (full 3×3 convolutions, more capacity than LiteUNetDS)
# ---------------------------------------------------------------------------

class ConvBnRelu(nn.Module):
    """Conv2d(3×3) → BN → ReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DoubleConv(nn.Module):
    """Two ConvBnRelu blocks."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.c1 = ConvBnRelu(in_ch, out_ch)
        self.c2 = ConvBnRelu(out_ch, out_ch)

    def forward(self, x):
        return self.c2(self.c1(x))


class DownStd(nn.Module):
    """MaxPool2d(2) + DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool  = nn.MaxPool2d(2)
        self.block = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.block(self.pool(x))


class UpStd(nn.Module):
    """Bilinear upsample × 2 + concat skip + DoubleConv."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.block = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if x.shape[-2:] != skip.shape[-2:]:
            dh = skip.size(-2) - x.size(-2)
            dw = skip.size(-1) - x.size(-1)
            skip = skip[
                :, :,
                dh // 2 : skip.size(-2) - (dh - dh // 2),
                dw // 2 : skip.size(-1) - (dw - dw // 2),
            ]
        return self.block(torch.cat([skip, x], dim=1))


# ---------------------------------------------------------------------------
# Attention Gate
# ---------------------------------------------------------------------------

class AttentionGate(nn.Module):
    """
    Additive attention gate for skip connections (Oktay et al., 2018).
    g  : gating signal from decoder (coarser resolution)
    x  : skip feature from encoder  (finer resolution)
    """

    def __init__(self, g_ch: int, x_ch: int, inter_ch: int):
        super().__init__()
        self.Wg  = nn.Conv2d(g_ch,     inter_ch, kernel_size=1, bias=False)
        self.Wx  = nn.Conv2d(x_ch,     inter_ch, kernel_size=1, bias=False)
        self.psi = nn.Conv2d(inter_ch, 1,        kernel_size=1, bias=True)

    def forward(self, g, x):
        g_up  = F.interpolate(g, size=x.shape[-2:], mode="bilinear", align_corners=False)
        alpha = torch.sigmoid(self.psi(F.relu(self.Wg(g_up) + self.Wx(x), inplace=True)))
        return x * alpha


# ---------------------------------------------------------------------------
# Lightweight ASPP bottleneck
# ---------------------------------------------------------------------------

class LightASPP(nn.Module):
    """
    3-branch ASPP using depthwise-separable dilated convs.
    Branches: 1×1  |  DS-dilated r=6  |  DS-dilated r=12
    → fused back to in_ch via 1×1 conv.
    Uses GroupNorm (stable with small batch sizes).
    """

    def __init__(self, in_ch: int, branch_ch: int = 32):
        super().__init__()
        self.b0 = nn.Sequential(
            nn.Conv2d(in_ch, branch_ch, 1, bias=False),
            nn.GroupNorm(min(8, branch_ch), branch_ch),
            nn.ReLU(inplace=True),
        )
        self.b1 = self._ds_dilated(in_ch, branch_ch, dilation=6)
        self.b2 = self._ds_dilated(in_ch, branch_ch, dilation=12)
        self.fuse = nn.Sequential(
            nn.Conv2d(branch_ch * 3, in_ch, 1, bias=False),
            nn.GroupNorm(min(8, in_ch), in_ch),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _ds_dilated(in_ch: int, out_ch: int, dilation: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=dilation, dilation=dilation,
                      groups=in_ch, bias=False),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.fuse(torch.cat([self.b0(x), self.b1(x), self.b2(x)], dim=1))


# ---------------------------------------------------------------------------
# UNetV2  (UNet + AttentionGates + LightASPP + DeepSupervision)
# ---------------------------------------------------------------------------

class UNetV2(nn.Module):
    """
    Improved U-Net for face parsing.

    Changes over UNet:
      - LightASPP at the bottleneck: multi-scale context (tiny eyes → large hair)
      - Attention gates on all 4 skip connections: suppress background noise,
        focus on face regions; critical for very small classes (ears, inner mouth)
      - Deep supervision: auxiliary 1×1 heads at H/8 and H/4 decoder stages
        give stronger gradient signal through the encoder (important with 900 samples)
        These heads are only active during training; inference returns main logits only.

    base=23, dropout=0.3  →  ~1.797M params  (budget: 1.8M)

    Returns during training (deep_supervision=True):
        (main_logits, aux1_logits_H8, aux2_logits_H4)
    Returns during eval / deep_supervision=False:
        main_logits
    """

    def __init__(self, num_classes: int = 19, base: int = 23,
                 dropout: float = 0.3, deep_supervision: bool = True):
        super().__init__()
        c1, c2, c3, c4, c5 = base, base * 2, base * 4, base * 8, base * 8

        # Encoder (identical to UNet)
        self.in_block   = DoubleConv(3, c1)
        self.down1      = DownStd(c1, c2)
        self.down2      = DownStd(c2, c3)
        self.down3      = DownStd(c3, c4)
        self.bottleneck = DownStd(c4, c5)

        # ASPP at bottleneck
        self.aspp = LightASPP(c5, branch_ch=32)
        self.drop = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()

        # Attention gates  (g_ch, x_ch, inter_ch)
        self.ag3 = AttentionGate(c5, c4, max(c4 // 4, 8))   # 184,184,46
        self.ag2 = AttentionGate(c3, c3, max(c3 // 4, 8))   # 92, 92, 23
        self.ag1 = AttentionGate(c2, c2, max(c2 // 4, 8))   # 46, 46, 11
        self.ag0 = AttentionGate(c1, c1, max(c1 // 4, 8))   # 23, 23,  8

        # Decoder (identical to UNet)
        self.up3 = UpStd(c5, c4, c3)
        self.up2 = UpStd(c3, c3, c2)
        self.up1 = UpStd(c2, c2, c1)
        self.up0 = UpStd(c1, c1, c1)

        self.head = nn.Conv2d(c1, num_classes, kernel_size=1)

        # Auxiliary heads (training only)
        self.deep_supervision = deep_supervision
        if deep_supervision:
            self.aux_head1 = nn.Conv2d(c3, num_classes, kernel_size=1)  # after up3, H/8
            self.aux_head2 = nn.Conv2d(c2, num_classes, kernel_size=1)  # after up2, H/4

    def forward(self, x):
        x1 = self.in_block(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        xb = self.bottleneck(x4)
        xb = self.aspp(xb)
        xb = self.drop(xb)

        d3 = self.up3(xb, self.ag3(xb, x4))
        d2 = self.up2(d3, self.ag2(d3, x3))
        d1 = self.up1(d2, self.ag1(d2, x2))
        d0 = self.up0(d1, self.ag0(d1, x1))

        out = self.head(d0)

        if self.deep_supervision and self.training:
            # Return aux logits at their NATIVE resolution (H/8 and H/4).
            # Do NOT upsample here — computing FocalDiceLoss on full 512×512
            # tensors is 16–64× more expensive than at native scale.
            # The train loop downsamples masks to match instead.
            return out, self.aux_head1(d3), self.aux_head2(d2)
        return out


class LiteUNetV2(nn.Module):
    """
    Lite U-Net V2: depthwise-separable encoder/decoder + all UNetV2 improvements.

    Combines DSConv parameter efficiency with:
      - LightASPP at the bottleneck for multi-scale context
      - Attention gates on all 4 skip connections
      - Deep supervision at H/8 and H/4 decoder stages
      - Dropout2d after bottleneck

    DSConv uses ~4× fewer params per block than standard 3×3 conv, so we can
    run wider channels (base=48 vs base=23) for the same budget → more capacity.

    base=48, dropout=0.3  →  ~1.0M params
    """

    def __init__(self, num_classes: int = 19, base: int = 48,
                 dropout: float = 0.3, deep_supervision: bool = True):
        super().__init__()
        c1, c2, c3, c4, c5 = base, base * 2, base * 4, base * 8, base * 8
        aspp_ch = max(32, c5 // 8)

        # Encoder (DSConv-based)
        self.in_block   = ConvBlock(3, c1)
        self.down1      = Down(c1, c2)
        self.down2      = Down(c2, c3)
        self.down3      = Down(c3, c4)
        self.bottleneck = Down(c4, c5)

        # ASPP + dropout at bottleneck
        self.aspp = LightASPP(c5, branch_ch=aspp_ch)
        self.drop = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()

        # Attention gates  (g_ch, x_ch, inter_ch)
        self.ag3 = AttentionGate(c5, c4, max(c4 // 4, 8))
        self.ag2 = AttentionGate(c3, c3, max(c3 // 4, 8))
        self.ag1 = AttentionGate(c2, c2, max(c2 // 4, 8))
        self.ag0 = AttentionGate(c1, c1, max(c1 // 4, 8))

        # Decoder (DSConv-based)
        self.up3 = Up(c5, c4, c3)
        self.up2 = Up(c3, c3, c2)
        self.up1 = Up(c2, c2, c1)
        self.up0 = Up(c1, c1, c1)

        self.head = nn.Conv2d(c1, num_classes, kernel_size=1)

        # Auxiliary heads (training only)
        self.deep_supervision = deep_supervision
        if deep_supervision:
            self.aux_head1 = nn.Conv2d(c3, num_classes, kernel_size=1)  # after up3, H/8
            self.aux_head2 = nn.Conv2d(c2, num_classes, kernel_size=1)  # after up2, H/4

    def forward(self, x):
        x1 = self.in_block(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        xb = self.bottleneck(x4)
        xb = self.aspp(xb)
        xb = self.drop(xb)

        d3 = self.up3(xb, self.ag3(xb, x4))
        d2 = self.up2(d3, self.ag2(d3, x3))
        d1 = self.up1(d2, self.ag1(d2, x2))
        d0 = self.up0(d1, self.ag0(d1, x1))

        out = self.head(d0)

        if self.deep_supervision and self.training:
            return out, self.aux_head1(d3), self.aux_head2(d2)
        return out


class UNet(nn.Module):
    """
    Standard U-Net for face parsing.
    Uses full 3×3 conv blocks (vs depthwise-separable in LiteUNetDS),
    giving more representational capacity at the cost of more parameters.

    base=23  →  ~1.73M params  (budget: 1,821,085)
    dropout  →  Dropout2d applied after bottleneck to regularise the
                deepest feature map and reduce overfitting on small datasets.
    """

    def __init__(self, num_classes: int = 19, base: int = 32, dropout: float = 0.0):
        super().__init__()
        c1, c2, c3, c4, c5 = base, base * 2, base * 4, base * 8, base * 8

        # Encoder
        self.in_block   = DoubleConv(3, c1)
        self.down1      = DownStd(c1, c2)
        self.down2      = DownStd(c2, c3)
        self.down3      = DownStd(c3, c4)
        self.bottleneck = DownStd(c4, c5)
        self.drop       = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()

        # Decoder (mirrors LiteUNetDS skip layout)
        self.up3 = UpStd(c5, c4, c3)
        self.up2 = UpStd(c3, c3, c2)
        self.up1 = UpStd(c2, c2, c1)
        self.up0 = UpStd(c1, c1, c1)

        self.head = nn.Conv2d(c1, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.in_block(x)    # (B, c1, H,    W)
        x2 = self.down1(x1)      # (B, c2, H/2,  W/2)
        x3 = self.down2(x2)      # (B, c3, H/4,  W/4)
        x4 = self.down3(x3)      # (B, c4, H/8,  W/8)
        xb = self.bottleneck(x4) # (B, c5, H/16, W/16)
        xb = self.drop(xb)       # regularise bottleneck (no-op at eval time)

        x = self.up3(xb, x4)    # -> c3
        x = self.up2(x, x3)     # -> c2
        x = self.up1(x, x2)     # -> c1
        x = self.up0(x, x1)     # -> c1

        return self.head(x)     # (B, num_classes, H, W)
