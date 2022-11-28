import math
import numpy as np
import torch
import torch.nn as nn

from attention import SpatialTransformer


class ModulatedLayerNorm(nn.Module):
    def __init__(
        self,
        num_features,
        eps=1e-6,
        channels_first=True,
    ):
        super().__init__()
        self.ln = nn.LayerNorm(num_features, eps=eps)
        self.gamma = nn.Parameter(torch.randn(1, 1, 1))
        self.beta = nn.Parameter(torch.randn(1, 1, 1))
        self.channels_first = channels_first

    def forward(
        self,
        x,
        w=None,
    ):
        x = x.permute(0, 2, 3, 1) if self.channels_first else x
        if w is None:
            x = self.ln(x)
        else:
            x = self.gamma * w * self.ln(x) + self.beta * w
        x = x.permute(0, 3, 1, 2) if self.channels_first else x
        return x


class ResBlock(nn.Module):
    def __init__(
        self,
        c,
        c_hidden,
        c_cond=0,
        c_skip=0,
        scaler=None,
        layer_scale_init_value=1e-6,
        c_cond_override=False,
    ):
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(c, c, kernel_size=3, groups=c)
        )
        self.ln = ModulatedLayerNorm(c, channels_first=False)
        if c_cond_override is False:
            self.channelwise = nn.Sequential(
                nn.Linear(c + c_skip, c_hidden),
                # nn.GELU(),
                nn.Mish(),
                nn.Linear(c_hidden, c),
            )
        else:
            self.channelwise = nn.Sequential(
                nn.Linear((c + c_skip) * 2, c_hidden),
                nn.GELU(),
                nn.Linear(c_hidden, c),
            )
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(c), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

        self.scaler = scaler
        # if c_cond > 0 and c_cond_override is None:
        self.cond_mapper = nn.Linear(c_cond, c)
        # if c_cond_override is not None:
        #     self.cond_mapper = nn.Linear(c_cond_override, c)

    def forward(
        self,
        x,
        s=None,
        skip=None,
    ):
        res = x
        x = self.depthwise(x)
        if s is not None:
            if s.size(2) == s.size(3) == 1:
                s = s.expand(-1, -1, x.size(2), x.size(3))
            elif s.size(2) != x.size(2) or s.size(3) != x.size(3):
                s = nn.functional.interpolate(s, size=x.shape[-2:], mode="bilinear")
            s = self.cond_mapper(s.permute(0, 2, 3, 1))
            # s = self.cond_mapper(s.permute(0, 2, 3, 1))
            # if s.size(1) == s.size(2) == 1:
            #     s = s.expand(-1, x.size(2), x.size(3), -1)
        x = self.ln(x.permute(0, 2, 3, 1), s)
        if skip is not None:
            x = torch.cat([x, skip.permute(0, 2, 3, 1)], dim=-1)
        x = self.channelwise(x)
        x = self.gamma * x if self.gamma is not None else x
        x = res + x.permute(0, 3, 1, 2)
        if self.scaler is not None:
            x = self.scaler(x)
        return x


class DenoiseUNet(nn.Module):
    def __init__(
        self,
        num_labels,
        c_hidden=1280,
        c_clip=2048,
        c_r=64,
        down_levels=[4, 8, 16, 32],
        up_levels=[32, 16, 8, 4],
        model_channels=320,
        num_heads=8,
        transformer_depth=1,
        context_dim=2048,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.c_r = c_r
        self.down_levels = down_levels
        self.up_levels = up_levels
        c_levels = [c_hidden // (2**i) for i in reversed(range(len(down_levels)))]
        self.embedding = nn.Embedding(num_labels, c_levels[0])
        dim_head = model_channels // num_heads

        # DOWN BLOCKS
        self.down_blocks = nn.ModuleList()
        for i, num_blocks in enumerate(down_levels):
            blocks = []


            if i > 0:
                blocks.append(
                    nn.Conv2d(
                        c_levels[i - 1],
                        c_levels[i],
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    )
                )
            for _ in range(num_blocks):
                block = ResBlock(
                    c_levels[i],
                    c_levels[i] * 4,
                    c_clip + c_r,
                )
                block.channelwise[-1].weight.data *= np.sqrt(1 / sum(down_levels))
                blocks.append(block)

                if i == 1:
                    blocks.append(SpatialTransformer(
                        model_channels,
                        num_heads,
                        dim_head,
                        depth=transformer_depth,
                        context_dim=context_dim,
                    ))

            self.down_blocks.append(nn.ModuleList(blocks))


        # UP BLOCKS
        self.up_blocks = nn.ModuleList()
        c_levels_up = [c_hidden // (2**i)
            for i in reversed(range(len(up_levels)))]
        c_levels_up.reverse()
        for i, num_blocks in enumerate(up_levels):
            blocks = []
            if i < len(c_levels_up) - 1:
                for j in range(num_blocks):
                    if i == len(c_levels_up) - 1:
                        block = SpatialTransformer(
                            model_channels,
                            num_heads,
                            dim_head,
                            depth=transformer_depth,
                            context_dim=context_dim,
                        )
                        blocks.append(block)

                    block = ResBlock(
                        c_levels_up[i],
                        c_levels_up[i] * 4,
                        (c_clip + c_r), # * 2,
                        c_levels_up[i] if (j == 0 and i > 0) else 0,
                        c_cond_override= False,#i == len(up_levels) - 1 and j == 0,
                    )
                    block.channelwise[-1].weight.data *= np.sqrt(1 / sum(c_levels_up))
                    blocks.append(block)
            if i != len(up_levels) - 1:
                blocks.append(
                    nn.ConvTranspose2d(
                        c_levels_up[i],
                        c_levels_up[i+1],
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    )
                )
            self.up_blocks.append(nn.ModuleList(blocks))

        self.clf = nn.Conv2d(c_levels[0], num_labels, kernel_size=1)

    def gamma(
        self,
        r,
    ):
        return (r * torch.pi / 2).cos()

    def add_noise(
        self,
        x,
        r,
        random_x=None,
    ):
        r = self.gamma(r)[:, None, None]
        mask = torch.bernoulli(
            r * torch.ones_like(x),
        )
        mask = mask.round().long()
        if random_x is None:
            random_x = torch.randint_like(x, 0, self.num_labels)
        x = x * (1 - mask) + random_x * mask
        return x, mask

    def gen_r_embedding(
        self,
        r,
        max_positions=10000,
    ):
        dtype = r.dtype
        r = self.gamma(r) * max_positions
        half_dim = self.c_r // 2
        emb = math.log(max_positions) / (half_dim - 1)
        emb = torch.arange(half_dim, device=r.device).float().mul(-emb).exp()
        emb = r[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        if self.c_r % 2 == 1:  # zero pad
            emb = nn.functional.pad(emb, (0, 1), mode="constant")
        return emb.to(dtype)

    def _down_encode_(
        self,
        x,
        s,
        c_full,
    ):
        level_outputs = []
        for i, blocks in enumerate(self.down_blocks):
            for j, block in enumerate(blocks):
                if isinstance(block, ResBlock):
                    x = block(x, s)
                elif isinstance(block, SpatialTransformer):
                    x = block(x, c_full)
                else:
                    x = block(x)
            level_outputs.insert(0, x)
        return level_outputs

    def _up_decode(
        self,
        level_outputs,
        s,
        c_full,
    ):
        x = level_outputs[0]
        for i, blocks in enumerate(self.up_blocks):
            for j, block in enumerate(blocks):
                if isinstance(block, ResBlock):
                    if i > 0 and j == 0:
                        x = block(x, s, level_outputs[i])
                    else:
                        x = block(x, s)
                elif isinstance(block, SpatialTransformer):
                    x = block(x, c_full)
                else:
                    if i == 1 and j == 0:
                        x_perm = x
                        x = block(x_perm)
                    else:
                        x = block(x)
        return x

    def forward(
        self,
        x,
        c,
        r,
        c_full,
    ):  # r is a uniform value between 0 and 1
        r_embed = self.gen_r_embedding(r)
        x = self.embedding(x).permute(0, 3, 1, 2)
        if len(c.shape) == 2:
            s = torch.cat([c, r_embed], dim=-1)[:, :, None, None]
        else:
            r_embed = r_embed[:, :, None, None].expand(-1, -1, c.size(2), c.size(3))
            s = torch.cat([c, r_embed], dim=1)
        level_outputs = self._down_encode_(x, s, c_full)
        x = self._up_decode(level_outputs, s, c_full)
        x = self.clf(x)
        return x


if __name__ == "__main__":
    device = "cuda"
    model = DenoiseUNet(2048).to(device)
    x_random = torch.randint(0, 2048, (1, 48, 48)).long().to(device)
    c_random = torch.randn((1, 2048)).to(device)
    r_random = torch.rand(1).to(device)
    c_full_random = torch.randn((1, 77, 2048)).to(device)
    model(x_random, c_random, r_random, c_full_random)
