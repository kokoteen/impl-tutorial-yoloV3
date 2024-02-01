import torch
from torch import nn


config = {
    "conv_01": (32, 3, 1),  # (filter, kernel_size, stride) padding=1 always
    "conv_02": (64, 3, 2),
    "res_01": 1,  # number of repetitions
    "conv_03": (128, 3, 2),
    "res_02": 2,
    "conv_04": (256, 3, 2),
    "res_03": 8,
    "conv_05": (512, 3, 2),
    "res_04": 8,
    "conv_06": (1024, 3, 2),
    "res_05": 4,
    "conv_07": (512, 1, 1),
    "conv_08": (1024, 3, 1),
    "conv_09": (512, 1, 1),
    "conv_10": (1024, 3, 1),
    "conv_11": (512, 1, 1),
    "pred_01": None,  # prediction layer (conv, conv+w/o bn)
    "conv_12": (256, 1, 1),
    "up_01": 2,  # upsample layer, stride=2
    "conv_13": (256, 1, 1),
    "conv_14": (512, 3, 1),
    "conv_15": (256, 1, 1),
    "conv_16": (512, 3, 1),
    "conv_17": (256, 1, 1),
    "pred_02": None,
    "conv_18": (128, 1, 1),
    "up_02": 2,
    "conv_19": (128, 1, 1),
    "conv_20": (256, 3, 1),
    "conv_21": (128, 1, 1),
    "conv_22": (256, 3, 1),
    "conv_23": (128, 1, 1),
    "pred_03": None,
}


class ConvolutionBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_bn: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            bias=False,
            **kwargs,
        )
        self.use_bn = use_bn
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
            return self.activation(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        num_repeats: int = 1,
    ):
        super().__init__()
        res_layers = [
            nn.Sequential(
                ConvolutionBlock(
                    in_channels=channels,
                    out_channels=channels // 2,
                    kernel_size=1,
                ),
                ConvolutionBlock(
                    in_channels=channels // 2,
                    out_channels=channels,
                    kernel_size=3,
                    padding=1,
                ),
            )
            for _ in range(num_repeats)
        ]

        self.layers = nn.ModuleList(res_layers)
        self.num_repeats = num_repeats

    def forward(self, x):
        for res_block in self.layers:
            x = x + res_block(x)
        return x


class ScalePrediction(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.pred = nn.Sequential(
            ConvolutionBlock(
                in_channels,
                in_channels * 2,
                kernel_size=3,
                padding=1,
            ),
            ConvolutionBlock(
                in_channels * 2,
                (num_classes + 5) * 3,  # why? 3 number of boxes per cell,
                # 5 = 4 coordinates + 1 objectness score
                use_bn=False,
                kernel_size=1,
            ),
        )
        self.num_classes = num_classes

    def forward(self, x):
        # kako znaju da su to te vrednosti
        return (
            self.pred(x)
            .reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3])
            .permute(0, 1, 3, 4, 2)
        )


class YOLOv3(nn.Module):
    def __init__(
        self,
        blocks: dict[dict[str, str]],
        in_channels: int = 3,
        num_classes: int = 80,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.layers = self._create_layers(blocks)

    def _create_layers(self, blocks: dict[dict[str, str]]):
        in_channels = self.in_channels
        module_list = nn.ModuleList()

        for layer in blocks:
            match layer.split(sep="_"):
                case ["conv", _]:
                    out_channels, kernel_size, stride = config[layer]
                    padding = 1 if kernel_size != 1 else 0
                    module_list.add_module(
                        name=layer,
                        module=ConvolutionBlock(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            use_bn=True,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding,
                        ),
                    )

                    in_channels = out_channels
                case ["res", _]:
                    num_repeats = config[layer]
                    module_list.add_module(
                        name=layer,
                        module=ResidualBlock(
                            channels=in_channels,
                            num_repeats=num_repeats,
                        ),
                    )
                case ["pred", _]:
                    module_list.add_module(
                        name=layer,
                        module=ScalePrediction(
                            in_channels=in_channels,
                            num_classes=self.num_classes,
                        ),
                    )
                case ["up", _]:
                    module_list.add_module(
                        name=layer,
                        module=nn.Upsample(scale_factor=config[layer]),
                    )
                    in_channels *= 3

        return module_list

    def forward(self, x):
        outputs = []
        res8_output = []
        for layer in self.layers:
            match layer:
                case ResidualBlock(num_repeats=8):
                    x = layer(x)
                    res8_output.append(x)
                case ConvolutionBlock() | ResidualBlock():
                    x = layer(x)
                case ScalePrediction():
                    outputs.append(layer(x))
                case nn.Upsample():
                    x = layer(x)
                    x = torch.cat([x, res8_output.pop()], dim=1)
        return outputs


if __name__ == "__main__":
    model = YOLOv3(blocks=config)
    num_classes = 80
    image_size = 416
    xx = torch.randn((2, 3, image_size, image_size))
    out = model(xx)
    assert out[0].shape == (2, 3, image_size // 32, image_size // 32, num_classes + 5)
