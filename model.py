import torch
import torch.nn as nn


def conv3x3(in_channels, out_channels, kernel_size, stride=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )


def dilated_conv_bn_act(in_channels, out_channels, act_fn, BatchNorm, dilation):
    model = nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            bias=False,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation,
        ),
        BatchNorm(out_channels),
        act_fn,
    )
    return model


def dilated_conv(in_channels, out_channels, kernel_size, dilation, stride=1):
    model = nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=dilation * (kernel_size // 2),
            dilation=dilation,
        )
    )
    return model


class ResidualBlockWithDilation(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        BatchNorm,
        kernel_size,
        stride=1,
        downsample=None,
        is_activation=True,
        is_top=False,
    ):
        super(ResidualBlockWithDilation, self).__init__()
        self.stride = stride
        self.downsample = downsample
        self.is_activation = is_activation
        self.is_top = is_top
        if self.stride != 1 or self.is_top:
            self.conv1 = conv3x3(in_channels, out_channels, kernel_size, self.stride)
            self.conv2 = conv3x3(out_channels, out_channels, kernel_size)
        else:
            self.conv1 = dilated_conv(in_channels, out_channels, kernel_size, dilation=3)
            self.conv2 = dilated_conv(out_channels, out_channels, kernel_size, dilation=3)

        self.bn1 = BatchNorm(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = BatchNorm(out_channels)

    def forward(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)

        out1 = self.relu(self.bn1(self.conv1(x)))
        out2 = self.bn2(self.conv2(out1))

        out2 += residual
        out = self.relu(out2)
        return out


class ResnetStraight(nn.Module):
    def __init__(
        self,
        num_filter,
        map_num,
        BatchNorm,
        block_nums=[3, 4, 6, 3],
        block=ResidualBlockWithDilation,
        kernel_size=5,
        stride=[1, 1, 2, 2],
    ):
        super(ResnetStraight, self).__init__()
        self.in_channels = num_filter * map_num[0]
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)
        self.block_nums = block_nums
        self.kernel_size = kernel_size

        self.layer1 = self.blocklayer(
            block,
            num_filter * map_num[0],
            self.block_nums[0],
            BatchNorm,
            kernel_size=self.kernel_size,
            stride=self.stride[0],
        )
        self.layer2 = self.blocklayer(
            block,
            num_filter * map_num[1],
            self.block_nums[1],
            BatchNorm,
            kernel_size=self.kernel_size,
            stride=self.stride[1],
        )
        self.layer3 = self.blocklayer(
            block,
            num_filter * map_num[2],
            self.block_nums[2],
            BatchNorm,
            kernel_size=self.kernel_size,
            stride=self.stride[2],
        )

    def blocklayer(self, block, out_channels, block_nums, BatchNorm, kernel_size, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(
                    self.in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                ),
                BatchNorm(out_channels),
            )

        layers = []
        layers.append(
            block(
                self.in_channels,
                out_channels,
                BatchNorm,
                kernel_size,
                stride,
                downsample,
                is_top=True,
            )
        )
        self.in_channels = out_channels
        for i in range(1, block_nums):
            layers.append(
                block(
                    out_channels,
                    out_channels,
                    BatchNorm,
                    kernel_size,
                    is_activation=True,
                    is_top=False,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        return out3


class UVDocnet(nn.Module):
    def __init__(self, num_filter, kernel_size=5):
        super(UVDocnet, self).__init__()
        self.num_filter = num_filter
        self.in_channels = 3
        self.kernel_size = kernel_size
        self.stride = [1, 2, 2, 2]

        BatchNorm = nn.BatchNorm2d
        act_fn = nn.ReLU(inplace=True)
        map_num = [1, 2, 4, 8, 16]

        self.resnet_head = nn.Sequential(
            nn.Conv2d(
                self.in_channels,
                self.num_filter * map_num[0],
                bias=False,
                kernel_size=self.kernel_size,
                stride=2,
                padding=self.kernel_size // 2,
            ),
            BatchNorm(self.num_filter * map_num[0]),
            act_fn,
            nn.Conv2d(
                self.num_filter * map_num[0],
                self.num_filter * map_num[0],
                bias=False,
                kernel_size=self.kernel_size,
                stride=2,
                padding=self.kernel_size // 2,
            ),
            BatchNorm(self.num_filter * map_num[0]),
            act_fn,
        )

        self.resnet_down = ResnetStraight(
            self.num_filter,
            map_num,
            BatchNorm,
            block_nums=[3, 4, 6, 3],
            block=ResidualBlockWithDilation,
            kernel_size=self.kernel_size,
            stride=self.stride,
        )

        map_num_i = 2
        self.bridge_1 = nn.Sequential(
            dilated_conv_bn_act(
                self.num_filter * map_num[map_num_i],
                self.num_filter * map_num[map_num_i],
                act_fn,
                BatchNorm,
                dilation=1,
            )
        )

        self.bridge_2 = nn.Sequential(
            dilated_conv_bn_act(
                self.num_filter * map_num[map_num_i],
                self.num_filter * map_num[map_num_i],
                act_fn,
                BatchNorm,
                dilation=2,
            )
        )

        self.bridge_3 = nn.Sequential(
            dilated_conv_bn_act(
                self.num_filter * map_num[map_num_i],
                self.num_filter * map_num[map_num_i],
                act_fn,
                BatchNorm,
                dilation=5,
            )
        )

        self.bridge_4 = nn.Sequential(
            *[
                dilated_conv_bn_act(
                    self.num_filter * map_num[map_num_i],
                    self.num_filter * map_num[map_num_i],
                    act_fn,
                    BatchNorm,
                    dilation=d,
                )
                for d in [8, 3, 2]
            ]
        )

        self.bridge_5 = nn.Sequential(
            *[
                dilated_conv_bn_act(
                    self.num_filter * map_num[map_num_i],
                    self.num_filter * map_num[map_num_i],
                    act_fn,
                    BatchNorm,
                    dilation=d,
                )
                for d in [12, 7, 4]
            ]
        )

        self.bridge_6 = nn.Sequential(
            *[
                dilated_conv_bn_act(
                    self.num_filter * map_num[map_num_i],
                    self.num_filter * map_num[map_num_i],
                    act_fn,
                    BatchNorm,
                    dilation=d,
                )
                for d in [18, 12, 6]
            ]
        )

        self.bridge_concat = nn.Sequential(
            nn.Conv2d(
                self.num_filter * map_num[map_num_i] * 6,
                self.num_filter * map_num[2],
                bias=False,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            BatchNorm(self.num_filter * map_num[2]),
            act_fn,
        )

        self.out_point_positions2D = nn.Sequential(
            nn.Conv2d(
                self.num_filter * map_num[2],
                self.num_filter * map_num[0],
                bias=False,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2,
                padding_mode="reflect",
            ),
            BatchNorm(self.num_filter * map_num[0]),
            nn.PReLU(),
            nn.Conv2d(
                self.num_filter * map_num[0],
                2,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2,
                padding_mode="reflect",
            ),
        )

        self.out_point_positions3D = nn.Sequential(
            nn.Conv2d(
                self.num_filter * map_num[2],
                self.num_filter * map_num[0],
                bias=False,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2,
                padding_mode="reflect",
            ),
            BatchNorm(self.num_filter * map_num[0]),
            nn.PReLU(),
            nn.Conv2d(
                self.num_filter * map_num[0],
                3,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2,
                padding_mode="reflect",
            ),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight, gain=0.2)
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                nn.init.xavier_normal_(m.weight, gain=0.2)

    def forward(self, x):
        resnet_head = self.resnet_head(x)
        resnet_down = self.resnet_down(resnet_head)
        bridge_1 = self.bridge_1(resnet_down)
        bridge_2 = self.bridge_2(resnet_down)
        bridge_3 = self.bridge_3(resnet_down)
        bridge_4 = self.bridge_4(resnet_down)
        bridge_5 = self.bridge_5(resnet_down)
        bridge_6 = self.bridge_6(resnet_down)
        bridge_concat = torch.cat([bridge_1, bridge_2, bridge_3, bridge_4, bridge_5, bridge_6], dim=1)
        bridge = self.bridge_concat(bridge_concat)

        out_point_positions2D = self.out_point_positions2D(bridge)
        out_point_positions3D = self.out_point_positions3D(bridge)

        return out_point_positions2D, out_point_positions3D
