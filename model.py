import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ELU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class NestNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_class=1, deep_supervision=False):
        super(NestNet, self).__init__()
        self.nb_filter = [32, 64, 128, 256, 512]

        self.down1 = DoubleConv(in_channels, self.nb_filter[0])
        self.pool1 = nn.MaxPool2d(2, 2)

        self.down2 = DoubleConv(self.nb_filter[0], self.nb_filter[1])
        self.pool2 = nn.MaxPool2d(2, 2)

        self.up1_2 = nn.ConvTranspose2d(self.nb_filter[1], self.nb_filter[0], kernel_size=2, stride=2)
        self.conv1_2 = DoubleConv(self.nb_filter[0]*2, self.nb_filter[0])

        self.down3 = DoubleConv(self.nb_filter[1], self.nb_filter[2])
        self.pool3 = nn.MaxPool2d(2, 2)

        self.up2_2 = nn.ConvTranspose2d(self.nb_filter[2], self.nb_filter[1], kernel_size=2, stride=2)
        self.conv2_2 = DoubleConv(self.nb_filter[1]*2, self.nb_filter[1])

        self.up1_3 = nn.ConvTranspose2d(self.nb_filter[1], self.nb_filter[0], kernel_size=2, stride=2)
        self.conv1_3 = DoubleConv(self.nb_filter[0]*3, self.nb_filter[0])

        self.down4 = DoubleConv(self.nb_filter[2], self.nb_filter[3])
        self.pool4 = nn.MaxPool2d(2, 2)

        self.up3_2 = nn.ConvTranspose2d(self.nb_filter[3], self.nb_filter[2], kernel_size=2, stride=2)
        self.conv3_2 = DoubleConv(self.nb_filter[2]*2, self.nb_filter[2])

        self.up2_3 = nn.ConvTranspose2d(self.nb_filter[2], self.nb_filter[1], kernel_size=2, stride=2)
        self.conv2_3 = DoubleConv(self.nb_filter[1]*3, self.nb_filter[1])

        self.up1_4 = nn.ConvTranspose2d(self.nb_filter[2], self.nb_filter[0], kernel_size=2, stride=2)
        self.conv1_4 = DoubleConv(self.nb_filter[0]*4, self.nb_filter[0])

        self.down5 = DoubleConv(self.nb_filter[3], self.nb_filter[4])

        self.up4_2 = nn.ConvTranspose2d(self.nb_filter[4], self.nb_filter[3], kernel_size=2, stride=2)
        self.conv4_2 = DoubleConv(self.nb_filter[3]*2, self.nb_filter[3])

        self.up3_3 = nn.ConvTranspose2d(self.nb_filter[3], self.nb_filter[2], kernel_size=2, stride=2)
        self.conv3_3 = DoubleConv(self.nb_filter[2]*3, self.nb_filter[2])

        self.up2_4 = nn.ConvTranspose2d(self.nb_filter[3], self.nb_filter[1], kernel_size=2, stride=2)
        self.conv2_4 = DoubleConv(self.nb_filter[1]*4, self.nb_filter[1])

        self.up1_5 = nn.ConvTranspose2d(self.nb_filter[1], self.nb_filter[0], kernel_size=2, stride=2)
        self.conv1_5 = DoubleConv(self.nb_filter[0]*5, self.nb_filter[0])

        self.output_1 = nn.Conv2d(self.nb_filter[0], num_class, kernel_size=1, padding=0, bias=True)
        self.output_2 = nn.Conv2d(self.nb_filter[0], num_class, kernel_size=1, padding=0, bias=True)
        self.output_3 = nn.Conv2d(self.nb_filter[0], num_class, kernel_size=1, padding=0, bias=True)
        self.output_4 = nn.Conv2d(self.nb_filter[0], num_class, kernel_size=1, padding=0, bias=True)

        self.deep_supervision = deep_supervision

    def forward(self, x):
        conv1_1 = self.down1(x)
        pool1 = self.pool1(conv1_1)

        conv2_1 = self.down2(pool1)
        pool2 = self.pool2(conv2_1)

        up1_2 = self.up1_2(conv2_1)
        merge1_2 = torch.cat((up1_2, conv1_1), dim=1)
        conv1_2 = self.conv1_2(merge1_2)

        conv3_1 = self.down3(pool2)
        pool3 = self.pool3(conv3_1)

        up2_2 = self.up2_2(conv3_1)
        merge2_2 = torch.cat((up2_2, conv2_1), dim=1)
        conv2_2 = self.conv2_2(merge2_2)

        up1_3 = self.up1_3(conv2_2)
        merge1_3 = torch.cat((up1_3, conv1_1, conv1_2), dim=1)
        conv1_3 = self.conv1_3(merge1_3)

        conv4_1 = self.down4(pool3)
        pool4 = self.pool4(conv4_1)

        up3_2 = self.up3_2(conv4_1)
        merge3_2 = torch.cat((up3_2, conv3_1), dim=1)
        conv3_2 = self.conv3_2(merge3_2)

        up2_3 = self.up2_3(conv3_2)
        merge2_3 = torch.cat((up2_3, conv2_1, conv2_2), dim=1)
        conv2_3 = self.conv2_3(merge2_3)

        up1_4 = self.up1_4(conv2_3)
        merge1_4 = torch.cat((up1_4, conv1_1, conv1_2, conv1_3), dim=1)
        conv1_4 = self.conv1_4(merge1_4)

        conv5_1 = self.down5(pool4)

        up4_2 = self.up4_2(conv5_1)
        merge4_2 = torch.cat((up4_2, conv4_1), dim=1)
        conv4_2 = self.conv4_2(merge4_2)

        up3_3 = self.up3_3(conv4_2)
        merge3_3 = torch.cat((up3_3, conv3_1, conv3_2), dim=1)
        conv3_3 = self.conv3_3(merge3_3)

        up2_4 = self.up2_4(conv3_3)
        merge2_4 = torch.cat((up2_4, conv2_1, conv2_2, conv2_3), dim=1)
        conv2_4 = self.conv2_4(merge2_4)

        up1_5 = self.up1_5(conv2_4)
        merge1_5 = torch.cat((up1_5, conv1_1, conv1_2, conv1_3, conv1_4), dim=1)
        conv1_5 = self.conv1_5(merge1_5)

        output_1 = self.output_1(conv1_2)
        output_2 = self.output_2(conv1_3)
        output_3 = self.output_3(conv1_4)
        output_4 = self.output_4(conv1_5)

        if self.deep_supervision:
            return [output_1, output_2, output_3, output_4]
        else:
            return output_4

# Example usage
input_channels = 1
output_channels = 1
model = NestNet(input_channels, output_channels)
print(model)
