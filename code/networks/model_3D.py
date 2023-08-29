import math
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchstat import stat

class UnetConv3(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, kernel_size=(3,3,1), padding_size=(1,1,0), init_stride=(1,1,1)):
        super(UnetConv3, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, init_stride, padding_size),
                                       nn.InstanceNorm3d(out_size),
                                       nn.ReLU(inplace=True),)
            self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                                       nn.InstanceNorm3d(out_size),
                                       nn.ReLU(inplace=True),)
        else:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, init_stride, padding_size),
                                       nn.ReLU(inplace=True),)
            self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                                       nn.ReLU(inplace=True),)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class REBNCONV(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1):
        super(REBNCONV, self).__init__()

        self.conv_s1 = nn.Conv3d(in_ch, out_ch, 3, padding=1 * dirate, dilation=1 * dirate)
        self.bn_s1 = nn.BatchNorm3d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x):
        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

        return xout


class UnetUp3(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, is_batchnorm=True):
        super(UnetUp3, self).__init__()
        if is_deconv:
            self.conv = UnetConv3(in_size, out_size, is_batchnorm)
            self.up = nn.ConvTranspose3d(in_size, out_size, kernel_size=(4,4,1), stride=(2,2,1), padding=(1,1,0))
        else:
            self.conv = UnetConv3(in_size+out_size, out_size, is_batchnorm)
            self.up = nn.Upsample(scale_factor=(2, 2, 1), mode='trilinear')

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2, 0]
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))


class UnetUp3_CT(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm=True):
        super(UnetUp3_CT, self).__init__()
        self.conv = UnetConv3(in_size + out_size, out_size, is_batchnorm, kernel_size=(3,3,3), padding_size=(1,1,1))
        self.up = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear')


    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2, 0]
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))


def _upsample_like(src, tar):
    # src = F.upsample(src, size=tar.shape[2:], mode='bilinear') # old version torch
    src = F.interpolate(src, size=tar.shape[2:], mode='trilinear', align_corners=True)
    return src

class RG1(nn.Module):  

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RG1, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2, ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2, ceil_mode=True)

        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.avg_pooling = torch.nn.AdaptiveMaxPool3d((1, 1, 1), return_indices=False)

        self.rebnconv7 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv8 = REBNCONV(mid_ch, mid_ch, dirate=2)


        self.rebnconv6d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)  


        hx1 = self.rebnconv1(hxin)  
        hx = self.pool1(hx1)  

        hx2 = self.rebnconv2(hx)  
        hx = self.pool2(hx2) 

        hx3 = self.rebnconv3(hx)  
        hx = self.pool3(hx3)  

        hx4 = self.rebnconv4(hx)  
        hx = self.pool4(hx4)  

        hx5 = self.rebnconv5(hx)  
        hx = self.pool5(hx5)  

        hx6 = self.rebnconv6(hx)  
        hx = self.avg_pooling(hx6)  # torch.Size([4, 8, 1, 1, 1])

        hx7 = self.rebnconv7(hx)  


        hx7 = _upsample_like(hx7, hx6)
        hx7 = self.rebnconv8(hx7)

        hx6d = self.rebnconv6d(torch.cat((hx7, hx6), 1))
        hx6dup = _upsample_like(hx6d, hx5)

        hx5d = self.rebnconv5d(torch.cat((hx6dup, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


class RG2(nn.Module):  

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RG2, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2, ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.avg_pooling = torch.nn.AdaptiveMaxPool3d((1, 1, 1), return_indices=False)

        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv7 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)


    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)
        hx = self.avg_pooling(hx5)

        hx6 = self.rebnconv6(hx)
        hx6 = _upsample_like(hx6, hx5)
        hx6 = self.rebnconv7(hx6)

        hx5d = self.rebnconv5d(torch.cat((hx6, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin





class models_3D(nn.Module):

    def __init__(self, feature_scale=4, in_channels=1, n_classes=1, is_deconv=True, is_batchnorm=True):
        super(models_3D, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = RG1(1, 8, 16)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv2 = RG2(16, 8, 32)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv3 = UnetConv3(filters[1], filters[2], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv4 = UnetConv3(filters[2], filters[3], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.center = UnetConv3(filters[3], filters[4], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))

        # upsampling
        self.up_concat4 = UnetUp3_CT(filters[4], filters[3], is_batchnorm)
        self.up_concat3 = UnetUp3_CT(filters[3], filters[2], is_batchnorm)
        self.up_concat2 = UnetUp3_CT(filters[2], filters[1], is_batchnorm)
        self.up_concat1 = UnetUp3_CT(filters[1], filters[0], is_batchnorm)

        # final conv (without any concat)
        self.final = nn.Conv3d(filters[0], n_classes, 1)

        self.out_conv1 = nn.Conv3d(filters[1], n_classes, 1, padding=0)
        self.out_conv2 = nn.Conv3d(filters[2], n_classes, 1, padding=0)
        self.out_conv3 = nn.Conv3d(filters[3], n_classes, 1, padding=0)
        self.out_conv4 = nn.Conv3d(filters[4], n_classes, 1, padding=0)

        self.out_conv5 = nn.Conv3d(n_classes * 5, n_classes, 1, padding=0)

        self.out_conv6 = nn.Conv3d(n_classes * 2, n_classes, 1, padding=0)
        self.out_conv7 = nn.Conv3d(n_classes * 2, n_classes, 1, padding=0)

        self.out_conv8 = nn.Conv3d(filters[0], n_classes, 1, padding=0)
        self.out_conv9 = nn.Conv3d(filters[1], n_classes, 1, padding=0)

        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.3)


    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)
        up5_out = self.out_conv4(center)

        center = self.dropout1(center)

        up4 = self.up_concat4(conv4, center)
        up4_out = self.out_conv3(up4)

        up3 = self.up_concat3(conv3, up4)
        up3_out = self.out_conv2(up3)

        up2 = self.up_concat2(conv2, up3)
        up2_out = self.out_conv1(up2)

        up1 = self.up_concat1(conv1, up2)

        up1 = self.dropout2(up1)

        final = self.final(up1)

        x2_out = _upsample_like(up2_out, final)
        x3_out = _upsample_like(up3_out, final)
        x4_out = _upsample_like(up4_out, final)
        x5_out = _upsample_like(up5_out, final)

        x1_bd = self.out_conv8(conv1)
        x2_bd = _upsample_like(conv2, x1_bd)
        x2_bd = self.out_conv9(x2_bd)
        x_bd = self.out_conv6(torch.cat((x1_bd, x2_bd), 1))
        final1 = self.out_conv5(torch.cat((final, x2_out, x3_out, x4_out, x5_out), 1))
        return final1, final, x_bd



