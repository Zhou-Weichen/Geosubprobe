from __future__ import annotations

import torch
from torch import nn
from transformers import ViTMAEForPreTraining

from .utils import center_padding, get_2d_sincos_pos_embed, tokens_to_output


class MAE(nn.Module):
    def __init__(
        self,
        checkpoint="facebook/vit-mae-base",
        output="dense",
        layer=-1,
        return_multilayer=False,
    ):
        """Code based on transformer database"""
        super().__init__()

        assert output in ["cls", "gap", "dense"], "Options: [cls, gap, dense]"
        self.output = output

        self.checkpoint_name = checkpoint.split("/")[1]

        self.vit = ViTMAEForPreTraining.from_pretrained(checkpoint).vit
        self.vit = self.vit.eval()

        # resize pos embedding
        # resize embedding for new size
        patch_size = self.vit.config.patch_size
        self.patch_size = patch_size
        self.layer = layer

        self.image_size = self.vit.embeddings.patch_embeddings.image_size
        self.feat_h = self.image_size[0] // self.patch_size
        self.feat_w = self.image_size[1] // self.patch_size

        feat_dim = self.vit.config.hidden_size
        num_layers = len(self.vit.encoder.layer)
        multilayers = [
            num_layers // 4 - 1,
            num_layers // 2 - 1,
            num_layers // 4 * 3 - 1,
            num_layers - 1,
        ]

        if return_multilayer:
            self.feat_dim = [feat_dim, feat_dim, feat_dim, feat_dim]
            self.multilayers = multilayers
        else:
            self.feat_dim = feat_dim
            layer = multilayers[-1] if layer == -1 else layer
            self.multilayers = [layer]

        # define layer name (for logging)
        self.layer = "-".join(str(_x) for _x in self.multilayers)

    def resize_pos_embed(self, image_size):
        # round up to a patch_size multiple so non-divisible inputs (e.g. KITTI
        # 350x1218 with patch=16) still get a valid grid; forward() pads images
        # to match via center_padding.
        h = (image_size[0] + self.patch_size - 1) // self.patch_size * self.patch_size
        w = (image_size[1] + self.patch_size - 1) // self.patch_size * self.patch_size
        image_size = (h, w)
        self.feat_h = image_size[0] // self.patch_size
        self.feat_w = image_size[1] // self.patch_size
        embed_dim = self.vit.config.hidden_size
        self.vit.embeddings.patch_embeddings.image_size = image_size
        pos_embed = get_2d_sincos_pos_embed(
            embed_dim, (self.feat_h, self.feat_w), add_cls_token=True
        )
        # there should be an easier way ... TODO
        device = self.vit.embeddings.patch_embeddings.projection.weight.device
        self.vit.embeddings.position_embeddings = nn.Parameter(
            torch.from_numpy(pos_embed).float().unsqueeze(0).to(device=device),
            requires_grad=False,
        )

    def embed_forward(self, embedder, pixel_values):
        # No masking here ...
        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = embedder.patch_embeddings(pixel_values)

        # add position embeddings w/o cls token
        embeddings = embeddings + embedder.position_embeddings[:, 1:, :]

        # append cls token
        cls_token = embedder.cls_token + embedder.position_embeddings[:, :1, :]
        cls_tokens = cls_token.expand(embeddings.shape[0], -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        return embeddings

    def forward(self, images):
        # pad to a patch_size multiple (matches dino.py / ibot.py)
        images = center_padding(images, self.patch_size)

        # check if positional embeddings are correct
        if self.image_size != images.shape[-2:]:
            self.resize_pos_embed(images.shape[-2:])

        # from MAE implementation
        head_mask = self.vit.get_head_mask(None, self.vit.config.num_hidden_layers)

        # ---- hidden ----
        # iterate encoder layers manually -- transformers >=4.57 dropped the
        # output_hidden_states / return_dict kwargs from ViTEncoder.forward,
        # and this matches the pattern in dino.py (forward) anyway.
        x = self.embed_forward(self.vit.embeddings, images)
        hidden_states = [x]
        for i, layer_mod in enumerate(self.vit.encoder.layer):
            layer_head_mask = head_mask[i] if head_mask is not None else None
            x = layer_mod(x, layer_head_mask)
            hidden_states.append(x)

        outputs = []
        for layer_i in self.multilayers:
            x_i = hidden_states[layer_i]
            x_i = tokens_to_output(
                self.output, x_i[:, 1:], x_i[:, 0], (self.feat_h, self.feat_w)
            )
            outputs.append(x_i)

        return outputs[0] if len(outputs) == 1 else outputs
