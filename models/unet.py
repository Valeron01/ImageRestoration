import typing

import torch
from torch import nn
from blocks import ResBlock, ConvBnReluBlock, DWConvBnReluBlock, SingleConvResBlock, ConvNextBlock


class UNet(nn.Module):
    def __init__(self, dim: int, depth: int, block: typing.Type = DWConvBnReluBlock):
        super().__init__()

        self.depth = depth

        self.stem = nn.Sequential(
            nn.Conv2d(3, dim, 7, 1, 3, bias=False),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(inplace=True)
        )

        self.encoder = nn.ModuleList([
            block(
                in_channels=2 ** current_depth * dim, out_channels=2 ** current_depth * dim * 2, stride=2
            ) for current_depth in range(depth)
        ])

        self.decoder = nn.ModuleList([
            # 2 * 2 is here because:
            # we need concatenate previous stages of encoder and decoder;
            # because of up-scaling
            nn.Sequential(
                block(
                    in_channels=2 * 2 ** current_depth * dim * (2 if current_depth != depth - 1 else 1),
                    out_channels=2 ** current_depth * dim
                ),
                nn.UpsamplingBilinear2d(scale_factor=2)
            ) for current_depth in reversed(range(depth))
        ])

        self.resulting_layer = nn.Conv2d(2 * dim, 3, 1)

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

        return self.resulting_layer(previous_stage)


if __name__ == '__main__':
    model = UNet(64, 6, block=DWConvBnReluBlock)
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()

    print(f"Params num is: {num_params / 1e6}M")


    print(model(torch.randn(1, 3, 256, 256)).shape)



    torch.onnx.export(model, torch.randn(1, 3, 256, 256), "./model.onnx", opset_version=17)
