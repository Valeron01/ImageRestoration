import typing

import torch
from torch import nn
from tqdm import trange

from blocks import ResBlock, ConvBnReluBlock, DWConvBnReluBlock, SingleConvResBlock, ConvNextBlock


class UNet(nn.Module):
    def __init__(self, n_features_list, block: typing.Type = DWConvBnReluBlock):
        super().__init__()
        self.n_features_list = n_features_list

        self.depth = len(n_features_list)

        self.stem = nn.Sequential(
            nn.Conv2d(3, n_features_list[0], 7, 1, 3, bias=False),
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
                nn.UpsamplingBilinear2d(scale_factor=2)
            ) for current_depth, (in_features, out_features) in enumerate(zip(reversed(n_features_list[1:]), reversed(n_features_list[:-1])))
        ])

        self.resulting_layer = nn.Conv2d(2 * n_features_list[0], 3, 1)

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
    model = UNet([64, 128, 256, 512, 1024], block=ConvNextBlock).cuda()
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print(f"Params num is: {num_params / 1e6}M")

    with torch.no_grad():
        for i in trange(1000):
            result = model(torch.randn(16, 3, 256, 256, device="cuda")).sum().item()

    # torch.onnx.export(model, torch.randn(1, 3, 256, 256), "./model.onnx", opset_version=17)
