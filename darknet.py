from torch import nn


def parse_cfg(cfgfile):
    """
    Takes a configuration file

    Returns a list of blocks. Each block describes a block in the neural
    network to be built. Block is represented as a dictionary in the list

    """
    with open(cfgfile, "r") as file:
        lines = file.read().split("\n")  # store the lines in a list
        lines = [x for x in lines if len(x) > 0]  # get read of the empty lines
        lines = [x for x in lines if x[0] != "#"]  # get rid of comments
        lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces

        block = {}
        blocks = []

        for line in lines:
            if line[0] == "[":  # This marks the start of a new block
                if (
                    len(block) != 0
                ):  # If block is not empty, implies it is storing values of previous block.
                    blocks.append(block)  # add it the blocks list
                    block = {}  # re-init the block
                block["type"] = line[1:-1].rstrip()
            else:
                key, value = line.split("=")
                block[key.rstrip()] = value.lstrip()
        blocks.append(block)

    return blocks


class CNNBlock(nn.Module):
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
        use_residual: bool = True,
        num_repeats: int = 1,
    ):
        super().__init__()
        res_layers = [
            nn.Sequential(
                CNNBlock(
                    in_channels=channels,
                    out_channels=channels // 2,
                    kernel_size=1,
                ),
                CNNBlock(
                    in_channels=channels // 2,
                    out_channels=channels,
                    kernel_size=3,
                    padding=1,
                ),
            )
            for _ in range(num_repeats)
        ]

        self.layers = nn.ModuleList(res_layers)
        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        if self.use_residual:
            for layer in self.layers:
                residual = x
                x = layer(x)
                x += residual
        else:
            for layer in self.layers:
                x = layer(x)

        return x


class ScalePrediction(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.pred = nn.Sequential(
            CNNBlock(
                in_channels,
                in_channels * 2,
                kernel_size=3,
                padding=1,
            ),
            CNNBlock(
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
    def __init__(self, in_channels: int = 3, num_classes: int = 80):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.layers = self._create_layers()

    def _create_layers(self):
        pass

    def forward(self, x):
        pass


if __name__ == "__main__":
    blocks = parse_cfg("./cfg/yolov3.cfg")
    print(blocks)
