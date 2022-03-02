# Adapted from: https://idiotdeveloper.com/unet-implementation-in-pytorch/

import torch
import torch.nn as nn

PADDING_MODE = "reflect"


class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        # By using padding we avoid reducing the size of the mask compared to original image
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, padding_mode=PADDING_MODE)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, padding_mode=PADDING_MODE)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x
    
class encoder_block(nn.Module):
    def __init__(self, in_c, out_c, proba_dropout):
        super().__init__()

        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))
        self.dropout = nn.Dropout(p=proba_dropout)

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        p = self.dropout(p)

        return x, p
    
class decoder_block(nn.Module):
    def __init__(self, in_c, out_c, proba_dropout):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c)
        self.dropout = nn.Dropout(p=proba_dropout)

    def forward(self, inputs, skip):
        inputs = self.dropout(inputs)
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)

        return x
    
class UNet(nn.Module):
    def __init__(self, proba_dropout, proba_dropout_middle):
        super().__init__()
        
        self.dropout = nn.Dropout(p=proba_dropout_middle)
        
        """ Encoder """
        self.e1 = encoder_block(3, 64, proba_dropout)
        self.e2 = encoder_block(64, 128, proba_dropout)
        self.e3 = encoder_block(128, 256, proba_dropout)
        self.e4 = encoder_block(256, 512, proba_dropout)

        """ Bottleneck """
        self.b = conv_block(512, 1024)

        """ Decoder """
        self.d1 = decoder_block(1024, 512, proba_dropout)
        self.d2 = decoder_block(512, 256, proba_dropout)
        self.d3 = decoder_block(256, 128, proba_dropout)
        self.d4 = decoder_block(128, 64, proba_dropout)

        """ Classifier """
        self.outputs = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        """ Bottleneck """
        b = self.b(p4)
        b = self.dropout(b)

        """ Decoder """
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        """ Classifier """
        outputs = self.outputs(d4)

        return outputs