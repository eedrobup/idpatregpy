# -*- coding: utf-8 -*-
"""
Example of simple 3 layer U-Net implementation

Variant of https://becominghuman.ai/implementing-unet-in-pytorch-8c7e05a121b4

Data flow:
    
     input - x1b ------------- x1c - x1d - output
              |                 |
             x2a - x2b - x2c - x2d
                    |     |
                   x3a - x3b

@author: Tim Cootes
"""

import torch
import torch.nn as nn


def two_conv(n_in_channel, n_out_channel):
    # Create two convolutional layers
    conv=nn.Sequential(
        nn.Conv2d(n_in_channel,n_out_channel,kernel_size=5,padding='same'),
        nn.ReLU(inplace=True),
        nn.Conv2d(n_out_channel,n_out_channel,kernel_size=5,padding='same'),
        nn.ReLU(inplace=True),
    )
    return conv

class UNet3(nn.Module):
    # Three level UNet 
    def __init__(self, nc1=16, nc2=32, nc3=64, n_out=8):
        super(UNet3,self).__init__()
        
        print("Setting up UNet3 to use ",nc1,nc2,nc3," channels.")
        
        # Downsampling side
        self.L_conv1=two_conv(3,nc1) 
        self.L_conv2=two_conv(nc1,nc2)
        self.L_conv3=two_conv(nc2,nc3)
        self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        
        # Upsampling side
        self.up_trans1 = nn.ConvTranspose2d(nc3,nc2, kernel_size=2, stride=2)
        self.R_conv1=two_conv(2*nc2,nc2)
        self.up_trans2 = nn.ConvTranspose2d(nc2,nc1, kernel_size=2, stride=2)
        self.R_conv2=two_conv(2*nc1,nc1)

        self.out = nn.Conv2d(nc1,n_out,kernel_size=1)        
        
    def forward(self,image):
        # Down sampling side
        x1b=self.L_conv1(image)
        
        x2a=self.maxpool(x1b)
        x2b=self.L_conv2(x2a)
        
        x3a=self.maxpool(x2b)
        x3b=self.L_conv3(x3a)
        
        # Upsampling side
        x2c=self.up_trans1(x3b)
        
        x2d=self.R_conv1(torch.cat([x2b,x2c],1))
        x1c=self.up_trans2(x2d)
        
        x1d=self.R_conv2(torch.cat([x1b,x1c],1))
        self.out(x1d)
        return self.out(x1d)


if __name__ == '__main__':
    image=torch.rand((1,1,72,80))
    model = UNet3()
    y=model(image)
    print("Input: ",image.shape)    
    print("Output: ",y.shape)
    if (image.shape[2]==y.shape[2] and image.shape[3]==y.shape[3] ):
        print("Input image size matches output")
    else:
        print("Problem! Output image is not the same size as input")