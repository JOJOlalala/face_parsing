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


class UNet(nn.Module):
    """
    Standard U-Net for face parsing.
    Uses full 3×3 conv blocks (vs depthwise-separable in LiteUNetDS),
    giving more representational capacity at the cost of more parameters.

    base=32  →  ~3.35M params   (LiteUNetDS base=48 ≈ 876K)
    """

    def __init__(self, num_classes: int = 19, base: int = 32):
        super().__init__()
        c1, c2, c3, c4, c5 = base, base * 2, base * 4, base * 8, base * 8

        # Encoder
        self.in_block   = DoubleConv(3, c1)
        self.down1      = DownStd(c1, c2)
        self.down2      = DownStd(c2, c3)
        self.down3      = DownStd(c3, c4)
        self.bottleneck = DownStd(c4, c5)

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

        x = self.up3(xb, x4)    # -> c3
        x = self.up2(x, x3)     # -> c2
        x = self.up1(x, x2)     # -> c1
        x = self.up0(x, x1)     # -> c1

        return self.head(x)     # (B, num_classes, H, W)
