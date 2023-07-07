import typing

import torch
from torch import nn


class UNet(nn.Module):
    def __init__(self, n_features_list, block: typing.Type,  in_channels: int = 3, out_channels: int = 3):
        super().__init__()
        self.n_features_list = n_features_list

        self.depth = len(n_features_list)

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, n_features_list[0], 7, 1, 3, bias=False),
            nn.BatchNorm2d(n_features_list[0]),
            nn.LeakyReLU(inplace=True)
        )

        self.encoder = nn.ModuleList([
            block(
                in_channels=in_features, out_channels=out_features, stride=2
            ) for in_features, out_features in zip(n_features_list[:-1], n_features_list[1:])
        ])

        self.decoder = nn.ModuleList([
            nn.Sequential(
                block(
                    in_channels=in_features * (1 if current_depth == 0 else 2),
                    out_channels=out_features
                ),
                # nn.ConvTranspose2d(out_features, out_features, 3, 2, padding=1, output_padding=1)
                nn.UpsamplingBilinear2d(scale_factor=2)
            ) for current_depth, (in_features, out_features) in enumerate(
                zip(reversed(n_features_list[1:]), reversed(n_features_list[:-1]))
            )
        ])

        self.resulting_layer = nn.Sequential(
            block(in_channels=n_features_list[0] * 2, out_channels=n_features_list[0]),
            nn.Conv2d(n_features_list[0], out_channels, 1)
        )

    def forward(self, x):
        stem = self.stem(x)

        downsample_stage = stem
        downsample_stages = [stem]
        for donwsample_layer in self.encoder:
            downsample_stage = donwsample_layer(downsample_stage)
            downsample_stages.append(downsample_stage)

        previous_stage = downsample_stages[-1]
        for encoder_features, upsample_layer in zip(reversed(downsample_stages[:-1]), self.decoder):
            new_stage = upsample_layer(previous_stage)
            previous_stage = torch.cat([encoder_features, new_stage], dim=1)

        return torch.tanh(self.resulting_layer(previous_stage))
