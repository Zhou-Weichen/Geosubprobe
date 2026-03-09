import torch
import torch.nn as nn
from torch.nn.functional import interpolate


class SurfaceNormalHead(nn.Module):
    def __init__(
        self,
        feat_dim,
        head_type="multiscale",
        uncertainty_aware=False,
        hidden_dim=512,
        kernel_size=1,
    ):
        super().__init__()

        self.uncertainty_aware = uncertainty_aware
        output_dim = 4 if uncertainty_aware else 3

        self.kernel_size = kernel_size

        assert head_type in ["linear", "multiscale", "dpt"]
        name = f"snorm_{head_type}_k{kernel_size}"
        self.name = f"{name}_UA" if uncertainty_aware else name

        if head_type == "linear":
            self.head = Linear(feat_dim, output_dim, kernel_size)
        elif head_type == "multiscale":
            self.head = MultiscaleHead(feat_dim, output_dim, hidden_dim, kernel_size)
        elif head_type == "dpt":
            self.head = DPT(feat_dim, output_dim, hidden_dim, kernel_size)
        else:
            raise ValueError(f"Unknown head type: {self.head_type}")

    def forward(self, feats):
        return self.head(feats)


class DepthHead(nn.Module):
    def __init__(
        self,
        feat_dim,
        head_type="multiscale",
        min_depth=0.001,
        max_depth=10,
        prediction_type="bindepth",
        hidden_dim=512,
        kernel_size=1,
        # conv1_weight=None,
        # conv2_weight=None,
        # conv2_bias=None,
    ):
        super().__init__()

        self.head_type = head_type
        self.kernel_size = kernel_size
        self.name = f"{prediction_type}_{head_type}_k{kernel_size}"

        if prediction_type == "bindepth":
            output_dim = 256
            self.predict = DepthBinPrediction(min_depth, max_depth, n_bins=output_dim)
        elif prediction_type == "sigdepth":
            output_dim = 1
            self.predict = DepthSigmoidPrediction(min_depth, max_depth)
        else:
            raise ValueError()

        if head_type == "linear":
            self.head = Linear(feat_dim, output_dim, kernel_size)
        elif head_type == "multiscale":
            self.head = MultiscaleHead(feat_dim, output_dim, hidden_dim, kernel_size)
        elif head_type == "dpt":
            self.head = DPT(feat_dim, output_dim, hidden_dim, kernel_size)
        elif head_type == 'dpt-s':
            self.head = SingleLayerDPT(feat_dim, hidden_dim, output_dim, kernel_size)
        elif head_type == "align2":
            self.head = LiteGAP8xDecoder(feat_dim[0], hidden_dim, output_dim)
        elif head_type == "align1":
            self.head = Linear_align(feat_dim[0], hidden_dim, output_dim) 
        elif head_type == "mlp":
            self.head = Litenonlinear(feat_dim, hidden_dim, output_dim, kernel_size)
        else:
            raise ValueError(f"Unknown head type: {self.head_type}")

    def forward(self, feats):
        """Prediction each pixel."""
        if self.head_type == "align1" or self.head_type == "align2" :
            pred_shallow, feats = self.head(feats)
            depth = self.predict(feats)
            return pred_shallow, depth
        else:
            feats = self.head(feats)
            depth = self.predict(feats)
            return depth


class DepthBinPrediction(nn.Module):
    def __init__(
        self,
        min_depth=0.001,
        max_depth=10,
        n_bins=256,
        bins_strategy="UD",
        norm_strategy="linear",
    ):
        super().__init__()
        self.n_bins = n_bins
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.norm_strategy = norm_strategy
        self.bins_strategy = bins_strategy

    def forward(self, prob):
        if self.bins_strategy == "UD":
            bins = torch.linspace(
                self.min_depth, self.max_depth, self.n_bins, device=prob.device
            )
        elif self.bins_strategy == "SID":
            bins = torch.logspace(
                self.min_depth, self.max_depth, self.n_bins, device=prob.device
            )

        # following Adabins, default linear
        if self.norm_strategy == "linear":
            prob = torch.relu(prob)
            eps = 0.1
            prob = prob + eps
            prob = prob / prob.sum(dim=1, keepdim=True)
        elif self.norm_strategy == "softmax":
            prob = torch.softmax(prob, dim=1)
        elif self.norm_strategy == "sigmoid":
            prob = torch.sigmoid(prob)
            prob = prob / prob.sum(dim=1, keepdim=True)

        depth = torch.einsum("ikhw,k->ihw", [prob, bins])
        depth = depth.unsqueeze(dim=1)
        return depth


class DepthSigmoidPrediction(nn.Module):
    def __init__(self, min_depth=0.001, max_depth=10):
        super().__init__()
        self.min_depth = min_depth
        self.max_depth = max_depth

    def forward(self, pred):
        depth = pred.sigmoid()
        depth = self.min_depth + depth * (self.max_depth - self.min_depth)
        return depth


class FeatureFusionBlock(nn.Module):
    def __init__(self, features, kernel_size, with_skip=True):
        super().__init__()
        self.with_skip = with_skip
        if self.with_skip:
            self.resConfUnit1 = ResidualConvUnit(features, kernel_size)

        self.resConfUnit2 = ResidualConvUnit(features, kernel_size)

    def forward(self, x, skip_x=None):
        if skip_x is not None:
            assert self.with_skip and skip_x.shape == x.shape
            x = self.resConfUnit1(x) + skip_x

        x = self.resConfUnit2(x)
        return x


class ResidualConvUnit(nn.Module):
    def __init__(self, features, kernel_size):
        super().__init__()
        assert kernel_size % 1 == 0, "Kernel size needs to be odd"
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(features, features, kernel_size, padding=padding),
            nn.ReLU(True),
            nn.Conv2d(features, features, kernel_size, padding=padding),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.conv(x) + x

class DPT(nn.Module):
    def __init__(self, input_dims, output_dim, hidden_dim=512, kernel_size=3):
        super().__init__()
        assert len(input_dims) == 4
        self.conv_0 = nn.Conv2d(input_dims[0], hidden_dim, 1, padding=0)
        self.conv_1 = nn.Conv2d(input_dims[1], hidden_dim, 1, padding=0)
        self.conv_2 = nn.Conv2d(input_dims[2], hidden_dim, 1, padding=0)
        self.conv_3 = nn.Conv2d(input_dims[3], hidden_dim, 1, padding=0)

        self.ref_0 = FeatureFusionBlock(hidden_dim, kernel_size)
        self.ref_1 = FeatureFusionBlock(hidden_dim, kernel_size)
        self.ref_2 = FeatureFusionBlock(hidden_dim, kernel_size)
        self.ref_3 = FeatureFusionBlock(hidden_dim, kernel_size, with_skip=False)

        self.out_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(hidden_dim, output_dim, 3, padding=1),
        )

    def forward(self, feats):
        """Prediction each pixel."""
        assert len(feats) == 4

        feats[0] = self.conv_0(feats[0])
        feats[1] = self.conv_1(feats[1])
        feats[2] = self.conv_2(feats[2])
        feats[3] = self.conv_3(feats[3])

        feats = [
            interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
            for x in feats
        ]

        out = self.ref_3(feats[3], None)
        out = self.ref_2(feats[2], out)
        out = self.ref_1(feats[1], out)
        out = self.ref_0(feats[0], out)

        out = interpolate(out, scale_factor=4, mode="bilinear", align_corners=True)
        out = self.out_conv(out)
        out = interpolate(out, scale_factor=2, mode="bilinear", align_corners=True)
        return out


def make_conv(input_dim, hidden_dim, output_dim, num_layers, kernel_size=1):
    if num_layers == 1:
        conv = nn.Conv2d(input_dim, output_dim, kernel_size)
    else:
        assert num_layers > 1
        modules = [nn.Conv2d(input_dim, hidden_dim, kernel_size), nn.ReLU(inplace=True)]
        for i in range(num_layers - 2):
            modules.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size))
            modules.append(nn.ReLU(inplace=True))
        modules.append(nn.Conv2d(hidden_dim, output_dim, kernel_size))
        conv = nn.Sequential(*modules)

    return conv


class Linear(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=1):
        super().__init__()
        if type(input_dim) is not int:
            input_dim = sum(input_dim)

        assert type(input_dim) is int
        padding = kernel_size // 2
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, padding=padding)

    def forward(self, feats):
        if type(feats) is list:
            feats = torch.cat(feats, dim=1)

        feats = interpolate(feats, scale_factor=4, mode="bilinear", align_corners=True)
        return self.conv(feats)

class Litenonlinear(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, kernel_size=1):
        super().__init__()
        if type(input_dim) is not int:
            input_dim = sum(input_dim)

        assert type(input_dim) is int
        padding = kernel_size // 2
        
        self.conv = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=1),
            nn.GroupNorm(1, hidden_dim), 
            nn.GELU(),
            nn.Conv2d(hidden_dim, output_dim, kernel_size=1),
        )
    def forward(self, feats):
        if type(feats) is list:
            feats = torch.cat(feats, dim=1)

        feats = interpolate(feats, scale_factor=4, mode="bilinear", align_corners=True)
        return self.conv(feats)
        



class MultiscaleHead(nn.Module):
    def __init__(self, input_dims, output_dim, hidden_dim=512, kernel_size=1):
        super().__init__()

        self.convs = nn.ModuleList(
            [make_conv(in_d, None, hidden_dim, 1, kernel_size) for in_d in input_dims]
        )
        interm_dim = len(input_dims) * hidden_dim
        self.conv_mid = make_conv(interm_dim, hidden_dim, hidden_dim, 3, kernel_size)
        self.conv_out = make_conv(hidden_dim, hidden_dim, output_dim, 2, kernel_size)

    def forward(self, feats):
        num_feats = len(feats)
        feats = [self.convs[i](feats[i]) for i in range(num_feats)]

        h, w = feats[-1].shape[-2:]
        feats = [
            interpolate(feat, (h, w), mode="bilinear", align_corners=True)
            for feat in feats
        ]
        feats = torch.cat(feats, dim=1).relu()

        # upsample
        feats = interpolate(feats, scale_factor=2, mode="bilinear", align_corners=True)
        feats = self.conv_mid(feats).relu()
        feats = interpolate(feats, scale_factor=4, mode="bilinear", align_corners=True)
        return self.conv_out(feats)




############################################## Single-scale DPT #####################################################

class SingleLayerDPT(nn.Module):
    def __init__(self, input_dims, hidden_dim, output_dim, kernel_size=3):
        super().__init__()
        self.proj = nn.Conv2d(input_dims, hidden_dim, 1, padding=0)

        self.refine = FeatureFusionBlock(hidden_dim,kernel_size,with_skip=False)

        self.out_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(hidden_dim, output_dim, 3, padding=1),
        )

    def forward(self, x):
        x = self.proj(x)
        x = interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        x = self.refine(x, None)
        x = interpolate(x, scale_factor=4, mode="bilinear", align_corners=True)
        x = self.out_conv(x)
        x = interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        return x


# class SingleLayerDPTWithAlign(nn.Module):
#     def __init__(self, input_dims, hidden_dim, output_dim, kernel_size=3):
#         super().__init__()
        
#         self.proj = nn.Conv2d(input_dims, hidden_dim, 1, padding=0)

#         self.pred_shallow_head = nn.Sequential(
#             nn.Conv2d(hidden_dim, input_dims//2, 3, padding=1),
#             nn.GELU(),
#         )

#         self.refine = FeatureFusionBlock(hidden_dim, kernel_size, with_skip=False)

#         self.out_conv = nn.Sequential(
#             nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
#             nn.ReLU(True),
#             nn.Conv2d(hidden_dim, output_dim, 3, padding=1),
#         )

#     def forward(self, x):
#         orig_h, orig_w = x.shape[-2:]
#         x = self.proj(x)
#         x = interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
#         x = self.refine(x, None)
#         x = interpolate(x, scale_factor=4, mode="bilinear", align_corners=True)
        
#         feat_shallow = self.pred_shallow_head(x)
#         pred_shallow = F.adaptive_avg_pool2d(feat_shallow, (orig_h, orig_w))
#         x = self.out_conv(x)
#         out = interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
#         return pred_shallow, out

# class Linear_align(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, kernel_size=1):
#         super().__init__()
#         if type(input_dim) is not int:
#             input_dim = sum(input_dim)

#         assert type(input_dim) is int
#         padding = kernel_size // 2
#         self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, padding=padding)
#         self.pred_shallow_head = nn.Sequential(
#             nn.Conv2d(output_dim, output_dim, 3, padding=1),
#             nn.GroupNorm(8, output_dim), 
#             nn.GELU(),
#             nn.Conv2d(output_dim, input_dim//2, 1, padding=1),
#         )
#     def forward(self, feats):
#         if type(feats) is list:
#             feats = torch.cat(feats, dim=1)
#         orig_h, orig_w = feats.shape[-2:]
#         feats = interpolate(feats, scale_factor=4, mode="bilinear", align_corners=True)
#         out = self.conv(feats)
#         pred_shallow = self.pred_shallow_head(out)
#         pred_shallow = F.adaptive_avg_pool2d(pred_shallow, (orig_h, orig_w))
#         return pred_shallow, out
###############################################################################################################

        
# class GeometryAwareProbe(nn.Module):
#     def __init__(self, in_channels, hidden_dim,output_dim):
#         super().__init__()
        
#         self.backbone = nn.Sequential(
#             nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
#             nn.GroupNorm(8, hidden_dim),
#             nn.ReLU(inplace=True),
        
#             nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
#             nn.GroupNorm(8, hidden_dim),
#             nn.ReLU(inplace=True),
#         )

#         # predict shallow spatial feature
#         self.shallow_head = nn.Conv2d(
#             hidden_dim, in_channels//2, kernel_size=1
#         )

#         # task head 
#         self.task_head = nn.Conv2d(
#             hidden_dim + in_channels//2, hidden_dim, kernel_size=3,padding=1)

#         self.out_conv = nn.Sequential(
#             nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
#             nn.ReLU(True),
#             nn.Conv2d(hidden_dim, output_dim, 3, padding=1),
#         )

        
#     def forward(self, deep_feat):
#         x = self.backbone(deep_feat)
#         pred_shallow = self.shallow_head(x)
#         fused = torch.cat([x, pred_shallow_up], dim=1)
#         pred_task = self.task_head(fused)
    
#         return pred_shallow, pred_task


# class GeometryAwareProbe(nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, mode='no_align'):
#         """
#         mode: 
#           - 'full': Complete model (with branches, splicing, and alignment loss)
#           - 'no_align': Ablation supervision (with branches and splicing, but no alignment loss is added during training)
#           - 'baseline': Ablation structure (no branches, no splicing, regressive ordinary convolution)
#         """
#         super().__init__()
#         self.mode = mode
        
#         self.backbone = nn.Sequential(
#             nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
#             nn.GroupNorm(8, hidden_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
#             nn.GroupNorm(8, hidden_channels),
#             nn.ReLU(inplace=True),
#         )

#         if self.mode != 'baseline':

#             self.alignment_head = nn.Conv2d(
#                 hidden_channels, in_channels // 2, kernel_size=1
#             )
#             self.task_head = nn.Conv2d(
#                 hidden_channels + in_channels // 2, out_channels, kernel_size=1
#             )
#         else:
#             self.task_head = nn.Conv2d(
#                 hidden_channels, out_channels, kernel_size=1
#             )

#     def forward(self, deep_feat):

#         latent = self.backbone(deep_feat)

#         if self.mode == 'baseline':
#             pred_task = self.task_head(latent)
#             return None, pred_task 

#         geo_features = self.alignment_head(latent)

#         fused = torch.cat([latent, geo_features], dim=1)
#         pred_task = self.task_head(fused)

#         return geo_features, pred_task


# class GeometryRefineUnit(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, 3, padding=1),
#             nn.GroupNorm(4, out_channels),
#             nn.GELU(), 
#             nn.Conv2d(out_channels, out_channels, 3, padding=1),
#             nn.GroupNorm(4, out_channels)
#         )
#         self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

#     def forward(self, x):
#         return F.gelu(self.shortcut(x) + self.conv(x))

# class GAP8xDecoder(nn.Module):
#     def __init__(self, in_channels, hidden_dim=256, out_channels=1):
#         super().__init__()
        
#         self.geo_probe = nn.Sequential(
#             nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
#             nn.GroupNorm(8, hidden_dim),
#             nn.GELU(),
#             nn.Conv2d(hidden_dim, in_channels // 2, kernel_size=1)
#         )

#         curr_dims = in_channels + (in_channels // 2)
        
#         self.up1 = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
#             GeometryRefineUnit(curr_dims, hidden_dim)
#         )
        
#         self.up2 = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
#             GeometryRefineUnit(hidden_dim, hidden_dim // 2)
#         )
        
#         self.up3 = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
#             GeometryRefineUnit(hidden_dim // 2, hidden_dim // 4)
#         )

#         self.final_head = nn.Conv2d(hidden_dim // 4, out_channels, kernel_size=3, padding=1)

#     def forward(self, feat):
#         geo_feat = self.geo_probe(feat) 
        
#         x = torch.cat([feat, geo_feat], dim=1)
        
#         x = self.up1(x) # -> 1/4 
#         x = self.up2(x) # -> 1/2 
#         x = self.up3(x) # -> 8x 
        
#         out = self.final_head(x)
        
#         return geo_feat, out
    
# ###############################################################################################################
# class LiteRefineUnit(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
#             nn.GroupNorm(min(in_channels, 8), in_channels),
#             nn.GELU(),
#             nn.Conv2d(in_channels, out_channels, 1, bias=False),
#             nn.GroupNorm(min(out_channels, 8), out_channels)
#         )
#         self.shortcut = nn.Conv2d(in_channels, out_channels, 1, bias=False) if in_channels != out_channels else nn.Identity()

#     def forward(self, x):
#         return F.gelu(self.shortcut(x) + self.conv(x))

# class LiteGAP8xDecoder(nn.Module):
#     def __init__(self, in_channels, hidden_dim=128, out_channels=1): 
#         super().__init__()
        
#         self.geo_probe = nn.Sequential(
#             nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
#             nn.GroupNorm(8, hidden_dim),
#             nn.GELU(),
#             nn.Conv2d(hidden_dim, in_channels//2, kernel_size=1) 
#         )

#         self.feature_proj = nn.Conv2d(in_channels, hidden_dim, 1, bias=False)

#         self.up1 = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
#             LiteRefineUnit(hidden_dim + in_channels//2, hidden_dim) # 1/4
#         )
        
#         self.up2 = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
#             LiteRefineUnit(hidden_dim, hidden_dim // 2) # 1/2
#         )
        
#         self.up3 = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
#             LiteRefineUnit(hidden_dim // 2, hidden_dim // 4) # 8x
#         )

#         self.final_head = nn.Conv2d(hidden_dim // 4, out_channels, kernel_size=1)

#     def forward(self, feat):
#         geo_feat = self.geo_probe(feat) 
#         x = self.feature_proj(feat)
        
#         x = torch.cat([x, geo_feat], dim=1)
        
#         x = self.up1(x) 
#         x = self.up2(x) 
#         x = self.up3(x) 
        
#         return geo_feat, self.final_head(x)

# ###############################################################################################################
# class RefineUnit(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.block = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
#             nn.GroupNorm(min(out_channels, 8), out_channels),
#             nn.GELU(),
#             nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
#             nn.GroupNorm(min(out_channels, 8), out_channels),
#         )
#         self.shortcut = (
#             nn.Conv2d(in_channels, out_channels, 1, bias=False)
#             if in_channels != out_channels else nn.Identity()
#         )

#     def forward(self, x):
#         return F.gelu(self.block(x) + self.shortcut(x))
        
# class UpBlock(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         self.up = nn.ConvTranspose2d(
#             in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False
#         )
#         self.norm = nn.GroupNorm(min(out_ch, 8), out_ch)
#         self.act = nn.GELU()
#         self.refine = RefineUnit(out_ch, out_ch)

#     def forward(self, x):
#         x = self.act(self.norm(self.up(x)))
#         return self.refine(x)
        
# class GeoEncoder(nn.Module):
#     def __init__(self, in_channels, geo_channels):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(in_channels, geo_channels, 1, bias=False),
#             nn.GroupNorm(min(geo_channels, 8), geo_channels),
#             nn.GELU(),
#             nn.Conv2d(geo_channels, geo_channels, 1, bias=False)
#         )

#     def forward(self, feat):
#         return self.net(feat)
        
# class GeoModulator(nn.Module):
#     def __init__(self, geo_channels, target_channels):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(geo_channels, target_channels * 2, 1, bias=False)
#         )

#     def forward(self, geo_feat):
#         gamma, beta = self.net(geo_feat).chunk(2, dim=1)
#         return gamma, beta

# class DisentangledGAP8xDecoder(nn.Module):
#     def __init__(
#         self,
#         in_channels,
#         hidden_dim=128,
#         out_channels=256
#     ):
#         super().__init__()

#         self.geo_encoder = GeoEncoder(in_channels, in_channels//2)

#         self.geo_mod1 = GeoModulator(in_channels//2, hidden_dim)
#         self.geo_mod2 = GeoModulator(in_channels//2, hidden_dim // 2)
#         self.geo_mod3 = GeoModulator(in_channels//2, hidden_dim // 4)

#         self.semantic_proj = nn.Sequential(
#             nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
#             nn.GroupNorm(8, hidden_dim),
#             nn.GELU()
#         )

#         self.up1 = UpBlock(hidden_dim, hidden_dim)      
#         self.up2 = UpBlock(hidden_dim, hidden_dim // 2)  
#         self.up3 = UpBlock(hidden_dim // 2, hidden_dim // 4)  

#         self.head = nn.Conv2d(hidden_dim // 4, out_channels, 1)

#     @staticmethod
#     def resize_like(src, ref):
#         return F.interpolate(
#             src,
#             size=ref.shape[-2:],
#             mode="bilinear",
#             align_corners=False
#         )
    
#     def forward(self, feat):
#         geo_feat = self.geo_encoder(feat)
#         x = self.semantic_proj(feat)
    
#         x = self.up1(x)
#         geo1 = self.resize_like(geo_feat, x)
#         gamma, beta = self.geo_mod1(geo1)
#         # print(x.shape)
#         # print(gamma.shape)
#         # print(beta.shape)
#         x = gamma * x + beta
    
#         x = self.up2(x)
#         geo2 = self.resize_like(geo_feat, x)
#         gamma, beta = self.geo_mod2(geo2)
#         x = gamma * x + beta
    
#         x = self.up3(x)
#         geo3 = self.resize_like(geo_feat, x)
#         gamma, beta = self.geo_mod3(geo3)
#         x = gamma * x + beta
    
#         out = self.head(x)
#         return geo_feat, out

# ###############################################################################################################

# class SharpSPD(nn.Module):
#     def __init__(self, in_dim=1024, rank_k=64, out_dim=1):
#         super().__init__()
        
#         self.bottleneck = nn.Sequential(
#             nn.Conv2d(in_dim, rank_k * 4, 1, bias=False),
#             nn.GroupNorm(8, rank_k * 4),
#             nn.GELU()
#         )
#         # PixelShuffle: 1/14 -> 1/7)
#         self.pixel_shuffle_up = nn.PixelShuffle(2) 

#         self.refine = nn.Sequential(
#             nn.Conv2d(rank_k, rank_k, 3, padding=1, groups=rank_k, bias=False),
#             nn.Conv2d(rank_k, rank_k, 1),
#             nn.GroupNorm(4, rank_k),
#             nn.GELU()
#         )

#         self.final_up = nn.Sequential(
#             nn.Upsample(scale_factor=7, mode='bilinear', align_corners=True),
#             nn.Conv2d(rank_k, rank_k // 2, 3, padding=1),
#             nn.GELU(),
#             nn.Conv2d(rank_k // 2, out_dim, 1)
#         )

#     def forward(self, x):
#         z_expanded = self.bottleneck(x)
#         z = self.pixel_shuffle_up(z_expanded)
#         z = self.refine(z)
#         out = self.final_up(z)
#         return z, out