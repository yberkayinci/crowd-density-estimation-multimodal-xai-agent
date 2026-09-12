import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ── Shared components ──────────────────────────────────────────

# First 3 pooling blocks of VGG (up to conv3_3 / conv3_4), output is H/8 x W/8
VGG16_FRONTEND_CFG = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
VGG19_FRONTEND_CFG = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512]

FRONTEND_CONFIGS = {
    'vgg16': VGG16_FRONTEND_CFG,
    'vgg19': VGG19_FRONTEND_CFG,
}


def make_frontend_layers(cfg, use_bn=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            conv = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if use_bn:
                layers.extend([conv, nn.BatchNorm2d(v), nn.ReLU(inplace=True)])
            else:
                layers.extend([conv, nn.ReLU(inplace=True)])
            in_channels = v
    return nn.Sequential(*layers)


def load_pretrained_vgg(frontend, backbone='vgg16'):
    """Load pretrained VGG weights into the frontend. Supports vgg16 and vgg19."""
    if backbone == 'vgg16':
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    elif backbone == 'vgg19':
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    own_convs = [m for m in frontend.modules() if isinstance(m, nn.Conv2d)]
    vgg_convs = [m for m in vgg.features.modules() if isinstance(m, nn.Conv2d)]
    for own, vgg_conv in zip(own_convs, vgg_convs):
        own.weight.data.copy_(vgg_conv.weight.data)
        if own.bias is not None and vgg_conv.bias is not None:
            own.bias.data.copy_(vgg_conv.bias.data)


def init_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


# ── Attention ──────────────────────────────────────────────────

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attention


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


# ── Backend helpers ────────────────────────────────────────────

def make_backend_layers(channels, dilation_rates, use_bn=False, dropout_rate=0.0):
    layers = []
    in_channels = 512
    for i, (out_ch, dil) in enumerate(zip(channels, dilation_rates)):
        conv = nn.Conv2d(in_channels, out_ch, kernel_size=3, padding=dil, dilation=dil)
        if use_bn:
            layers.extend([conv, nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)])
        else:
            layers.extend([conv, nn.ReLU(inplace=True)])
        if dropout_rate > 0 and i in [1, 3]:
            layers.append(nn.Dropout2d(p=dropout_rate))
        in_channels = out_ch
    return nn.Sequential(*layers)


# ── Models ─────────────────────────────────────────────────────

class CSRNetOriginal(nn.Module):
    def __init__(self, load_weights=True, backbone='vgg16'):
        super().__init__()
        self.backbone = backbone
        cfg = FRONTEND_CONFIGS[backbone]
        self.frontend = make_frontend_layers(cfg)
        backend_cfg = [512, 512, 512, 256, 128, 64]
        dilation_rates = [2, 2, 2, 2, 2, 2]
        self.backend = make_backend_layers(backend_cfg, dilation_rates)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        init_weights(self)
        if load_weights:
            load_pretrained_vgg(self.frontend, backbone)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return F.relu(x, inplace=True)


class CSRNetImproved(nn.Module):
    def __init__(
        self,
        load_weights=True,
        use_attention=True,
        use_bn=False,
        dropout_rate=0.0,
        backbone='vgg16',
        backend_channels=[512, 512, 512, 256, 128, 64],
        dilation_rates=[2, 2, 2, 2, 2, 2],
    ):
        super().__init__()
        self.use_attention = use_attention
        self.backbone = backbone
        cfg = FRONTEND_CONFIGS[backbone]

        self.frontend = make_frontend_layers(cfg, use_bn=use_bn)

        if use_attention:
            self.frontend_attention = CBAM(512, reduction=16)

        self.backend = make_backend_layers(
            backend_channels, dilation_rates,
            use_bn=use_bn, dropout_rate=dropout_rate,
        )

        if use_attention:
            self.backend_attention = CBAM(backend_channels[-1], reduction=8)

        self.output_layer = nn.Conv2d(backend_channels[-1], 1, kernel_size=1)

        init_weights(self)
        if load_weights:
            load_pretrained_vgg(self.frontend, backbone)

    def forward(self, x):
        x = self.frontend(x)
        if self.use_attention:
            x = self.frontend_attention(x)
        x = self.backend(x)
        if self.use_attention:
            x = self.backend_attention(x)
        x = self.output_layer(x)
        return F.relu(x, inplace=True)


class CSRNetMultiScale(nn.Module):
    def __init__(self, load_weights=True, use_attention=True, backbone='vgg16'):
        super().__init__()
        self.use_attention = use_attention
        self.backbone = backbone
        cfg = FRONTEND_CONFIGS[backbone]

        self.frontend = make_frontend_layers(cfg)

        backend_cfg = [512, 512, 512, 256, 128, 64]
        self.backend_d2 = make_backend_layers(backend_cfg, [2]*6)
        self.backend_d4 = make_backend_layers(backend_cfg, [4]*6)
        self.backend_d8 = make_backend_layers(backend_cfg, [8]*6)

        self.fusion = nn.Conv2d(64 * 3, 64, kernel_size=1)

        if use_attention:
            self.attention = CBAM(64, reduction=8)

        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        init_weights(self)
        if load_weights:
            load_pretrained_vgg(self.frontend, backbone)

    def forward(self, x):
        feat = self.frontend(x)
        out_d2 = self.backend_d2(feat)
        out_d4 = self.backend_d4(feat)
        out_d8 = self.backend_d8(feat)

        fused = self.fusion(torch.cat([out_d2, out_d4, out_d8], dim=1))
        if self.use_attention:
            fused = self.attention(fused)

        x = self.output_layer(fused)
        return F.relu(x, inplace=True)


# ── Factory ────────────────────────────────────────────────────

def create_model(model_type='improved', **kwargs):
    if model_type == 'original':
        valid = {'load_weights', 'backbone'}
        filtered = {k: v for k, v in kwargs.items() if k in valid}
        return CSRNetOriginal(**filtered)
    elif model_type == 'improved':
        return CSRNetImproved(**kwargs)
    elif model_type == 'multiscale':
        valid = {'load_weights', 'use_attention', 'backbone'}
        filtered = {k: v for k, v in kwargs.items() if k in valid}
        return CSRNetMultiScale(**filtered)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    print("Testing models...")
    x = torch.randn(2, 3, 256, 320)

    for name, mtype, kw in [
        ("Original VGG16", 'original', dict(load_weights=False)),
        ("Original VGG19", 'original', dict(load_weights=False, backbone='vgg19')),
        ("Improved VGG16", 'improved', dict(load_weights=False, use_attention=True)),
        ("Improved VGG19", 'improved', dict(load_weights=False, use_attention=True, backbone='vgg19')),
        ("MultiScale VGG16", 'multiscale', dict(load_weights=False, use_attention=True)),
    ]:
        model = create_model(mtype, **kw)
        y = model(x)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"{name}: {x.shape} -> {y.shape}, params={n_params:,}")
        assert (y >= 0).all(), f"{name} produced negative outputs!"

    print("All model tests passed!")
