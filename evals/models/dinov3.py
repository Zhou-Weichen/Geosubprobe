from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModel

from .utils import center_padding, tokens_to_output


class DINOv3(nn.Module):
    def __init__(
        self,
        checkpoint="facebook/dinov3-vitl16-pretrain-lvd1689m",
        output="dense",
        layer=-1,
        return_multilayer=False,
    ):
        super().__init__()

        assert output in ["cls", "gap", "dense", "dense-cls"]
        self.output = output

        self.vit = AutoModel.from_pretrained(checkpoint).eval()
        cfg = self.vit.config
        self.patch_size = cfg.patch_size
        self.num_register_tokens = cfg.num_register_tokens
        feat_dim = cfg.hidden_size
        num_layers = cfg.num_hidden_layers

        self.checkpoint_name = "dinov3_vitl16"

        feat_dim_out = feat_dim * 2 if output == "dense-cls" else feat_dim
        multilayers = [
            num_layers // 4 - 1,
            num_layers // 2 - 1,
            num_layers // 4 * 3 - 1,
            num_layers - 1,
        ]
        if return_multilayer:
            self.feat_dim = [feat_dim_out] * 4
            self.multilayers = multilayers
        else:
            self.feat_dim = feat_dim_out
            layer = multilayers[-1] if layer == -1 else layer
            self.multilayers = [layer]

        self.layer = "-".join(str(_x) for _x in self.multilayers)

    def forward(self, images):
        images = center_padding(images, self.patch_size)
        h = images.shape[-2] // self.patch_size
        w = images.shape[-1] // self.patch_size

        out = self.vit(pixel_values=images, output_hidden_states=True)
        hidden_states = out.hidden_states  # tuple len = num_layers + 1

        outputs = []
        for layer_i in self.multilayers:
            x_i = hidden_states[layer_i + 1]
            cls_tok = x_i[:, 0]
            spatial = x_i[:, 1 + self.num_register_tokens :]
            assert spatial.shape[1] == h * w, (spatial.shape, h, w)
            x_i = tokens_to_output(self.output, spatial, cls_tok, (h, w))
            outputs.append(x_i)

        return outputs[0] if len(outputs) == 1 else outputs
