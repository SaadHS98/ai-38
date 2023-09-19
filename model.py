import torch
import torch.nn as nn
import torch.nn.functional as F


class StandardUnit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(StandardUnit, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn(self.conv1(x)))
        x = F.relu(self.bn(self.conv2(x)))
        return x


class NestNet(nn.Module):
    def __init__(self, in_channels, num_classes, deep_supervision=False):
        super(NestNet, self).__init__()
        self.deep_supervision = deep_supervision

        nb_filter = [32, 64, 128, 256, 512]

        self.conv1_1 = StandardUnit(in_channels, nb_filter[0])
        self.pool1 = nn.MaxPool2d(2, stride=2)

        self.conv2_1 = StandardUnit(nb_filter[0], nb_filter[1])
        self.pool2 = nn.MaxPool2d(2, stride=2)

        self.up1_2 = nn.ConvTranspose2d(nb_filter[1], nb_filter[0], kernel_size=2, stride=2, padding=0)
        self.conv1_2 = StandardUnit(nb_filter[0] * 2, nb_filter[0])

        self.conv3_1 = StandardUnit(nb_filter[1], nb_filter[2])
        self.pool3 = nn.MaxPool2d(2, stride=2)

        self.up2_2 = nn.ConvTranspose2d(nb_filter[2], nb_filter[1], kernel_size=2, stride=2, padding=0)
        self.conv2_2 = StandardUnit(nb_filter[1] * 2, nb_filter[1])

        self.up1_3 = nn.ConvTranspose2d(nb_filter[1], nb_filter[0], kernel_size=2, stride=2, padding=0)
        self.conv1_3 = StandardUnit(nb_filter[0] * 3, nb_filter[0])

        self.conv4_1 = StandardUnit(nb_filter[2], nb_filter[3])
        self.pool4 = nn.MaxPool2d(2, stride=2)

        self.up3_2 = nn.ConvTranspose2d(nb_filter[3], nb_filter[2], kernel_size=2, stride=2, padding=0)
        self.conv3_2 = StandardUnit(nb_filter[2] * 2, nb_filter[2])

        self.up2_3 = nn.ConvTranspose2d(nb_filter[2], nb_filter[1], kernel_size=2, stride=2, padding=0)
        self.conv2_3 = StandardUnit(nb_filter[1] * 3, nb_filter[1])

        self.up1_4 = nn.ConvTranspose2d(nb_filter[1], nb_filter[0], kernel_size=2, stride=2, padding=0)
        self.conv1_4 = StandardUnit(nb_filter[0] * 4, nb_filter[0])

        self.conv5_1 = StandardUnit(nb_filter[3], nb_filter[4])

        self.up4_2 = nn.ConvTranspose2d(nb_filter[4], nb_filter[3], kernel_size=2, stride=2, padding=0)
        self.conv4_2 = StandardUnit(nb_filter[3] * 2, nb_filter[3])

        self.up3_3 = nn.ConvTranspose2d(nb_filter[3], nb_filter[2], kernel_size=2, stride=2, padding=0)
        self.conv3_3 = StandardUnit(nb_filter[2] * 3, nb_filter[2])

        self.up2_4 = nn.ConvTranspose2d(nb_filter[2], nb_filter[1], kernel_size=2, stride=2, padding=0)
        self.conv2_4 = StandardUnit(nb_filter[1] * 4, nb_filter[1])

        self.up1_5 = nn.ConvTranspose2d(nb_filter[1], nb_filter[0], kernel_size=2, stride=2, padding=0)
        self.conv1_5 = StandardUnit(nb_filter[0] * 5, nb_filter[0])

        self.nestnet_output_1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        self.nestnet_output_2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        self.nestnet_output_3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        self.nestnet_output_4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, x):
        conv1_1 = self.conv1_1(x)
        pool1 = self.pool1(conv1_1)

        conv2_1 = self.conv2_1(pool1)
        pool2 = self.pool2(conv2_1)

        up1_2 = self.up1_2(conv2_1)
        conv1_2 = self.conv1_2(torch.cat([up1_2, conv1_1], dim=1))

        conv3_1 = self.conv3_1(pool2)
        pool3 = self.pool3(conv3_1)

        up2_2 = self.up2_2(conv3_1)
        conv2_2 = self.conv2_2(torch.cat([up2_2, conv2_1], dim=1))

        up1_3 = self.up1_3(conv2_2)
        conv1_3 = self.conv1_3(torch.cat([up1_3, conv1_1, conv1_2], dim=1))

        conv4_1 = self.conv4_1(pool3)
        pool4 = self.pool4(conv4_1)

        up3_2 = self.up3_2(conv4_1)
        conv3_2 = self.conv3_2(torch.cat([up3_2, conv3_1], dim=1))

        up2_3 = self.up2_3(conv3_2)
        conv2_3 = self.conv2_3(torch.cat([up2_3, conv2_1, conv2_2], dim=1))

        up1_4 = self.up1_4(conv2_3)
        conv1_4 = self.conv1_4(torch.cat([up1_4, conv1_1, conv1_2, conv1_3], dim=1))

        conv5_1 = self.conv5_1(pool4)

        up4_2 = self.up4_2(conv5_1)
        conv4_2 = self.conv4_2(torch.cat([up4_2, conv4_1], dim=1))

        up3_3 = self.up3_3(conv4_2)
        conv3_3 = self.conv3_3(torch.cat([up3_3, conv3_1, conv3_2], dim=1))

        up2_4 = self.up2_4(conv3_3)
        conv2_4 = self.conv2_4(torch.cat([up2_4, conv2_1, conv2_2, conv2_3], dim=1))

        up1_5 = self.up1_5(conv2_4)
        conv1_5 = self.conv1_5(torch.cat([up1_5, conv1_1, conv1_2, conv1_3, conv1_4], dim=1))

        nestnet_output_1 = self.nestnet_output_1(conv1_2)
        nestnet_output_2 = self.nestnet_output_2(conv1_3)
        nestnet_output_3 = self.nestnet_output_3(conv1_4)
        nestnet_output_4 = self.nestnet_output_4(conv1_5)

        if self.deep_supervision:
            return nestnet_output_1, nestnet_output_2, nestnet_output_3, nestnet_output_4
        else:
            return nestnet_output_4


if __name__ == '__main__':
    model = NestNet(in_channels=1, num_classes=1)
    print(model)
