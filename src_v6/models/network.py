# ------------------------------------------------------------------------
# Copyright (c) 2023 CandleLabAI. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Network definition for Livenet
"""

# imports
from .blocks import GrayEncoder, ResidualEncoder, ResidualEncoder_6, DecoderBlock_S, BasicBlock, SingleDecoder, ResidualDecoder, RefineBlock, DecoderBlock
from .utils import get_final_image, combine_YCbCr_and_RGB
from torchvision.transforms import Resize
import torch.nn.functional as F
import torch.nn as nn
import torch


class Generator(torch.nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        self.opt = opt
        self.gray_encoder = GrayEncoder(in_nc=opt["gray"]["in_nc"])
        self.residual_encoder = ResidualEncoder(in_nc=opt["tranmission_map"]["in_nc"])
        self.gray_decoder = ResidualDecoder(out_nc=opt["gray"]["out_nc"])
        self.tm_decoder = ResidualDecoder(out_nc=opt["tranmission_map"]["out_nc"])
        self.atmos_decoder = ResidualDecoder(out_nc=opt["atmospheric_light"]["out_nc"])

    def forward(self, input_tensor):
        gfm_1, gfm_2, gfm_3, gfm_4, gfm_5 = self.gray_encoder(input_tensor)
        fm_1, fm_2, fm_3, fm_4, fm_5 = self.residual_encoder(input_tensor)
        gray = self.gray_decoder(gfm_5, gfm_4, gfm_3, gfm_2, gfm_1)
        tm = self.tm_decoder(fm_5, fm_4, fm_3, fm_2, fm_1)
        atmos = self.atmos_decoder(fm_5, fm_4, fm_3, fm_2, fm_1)
        atmos = F.avg_pool2d(atmos, (atmos.shape[2], atmos.shape[3]))
        atmos = atmos.view(atmos.shape[0], -1)
        coarsemap = get_final_image(input_tensor.detach(), atmos.detach(), tm.detach(), self.opt["tmin"])
        coarsemap, gray = combine_YCbCr_and_RGB(coarsemap, gray)
        return gray, tm, atmos, coarsemap


class Refiner(torch.nn.Module):
    def __init__(self, opt):
        super(Refiner, self).__init__()
        self.opt = opt
        self.residual6_encoder = ResidualEncoder_6(in_nc=opt["tranmission_map"]["in_nc"])
        self.single_decoder = SingleDecoder(out_nc=opt["tranmission_map"]["out_nc"])
        
        self.resize_256 = Resize((256, 256))
        self.resize_128 = Resize((128,128))
        self.conv_1x1 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.conv_3 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1, stride=1, padding=0)
        
    def upsampling(self, input_tensor, target_size):
        return F.interpolate(input_tensor, size=target_size, mode='bicubic')
        
    def gaussian_kernel(self, size, sigma):
        coords = torch.arange(size, dtype=torch.float32)
        coords -= size // 2
        
        g = coords**2
        g = (-g / (2 * sigma**2)).exp()
        
        g /= g.sum()
        return g.outer(g)
    
    def gaussian_blur(self, img, kernel_size=5, sigma=1.0):
        kernel = self.gaussian_kernel(kernel_size, sigma)
        kernel = kernel.expand(1, img.shape[1], kernel_size, kernel_size)
        kernel = kernel.to(img.device).type(img.dtype)
        
        padding = kernel_size // 2
        blurred_img = F.conv2d(img, kernel, padding=padding)
        
        return blurred_img
        
        
    def forward(self, coarsemap, gray):
        #원본 이미지 블러
        blurred_coarsemap = self.gaussian_blur(coarsemap, kernel_size=5, sigma=1.0)
        blurred_gray = self.gaussian_blur(gray, kernel_size=5, sigma=1.0)
        
        #원본 블러 이미지 -> 1/2 
        coarsemap_256 = self.resize_256(blurred_coarsemap)
        gray_256 = self.resize_256(blurred_gray)
        
        #1/2 이미지 블러
        blurred_coarsemap_256 = self.gaussian_blur(coarsemap_256, kernel_size=5, sigma=1.0)
        blurred_gray_256 = self.gaussian_blur(gray_256, kernel_size=5, sigma=1.0)
        
        #1/2 블러 이미지 -> 1/4
        coarsemap_128 = self.resize_128(blurred_coarsemap_256)
        gray_128 = self.resize_128(blurred_gray_256)
        
        #1/4 이미지 블러
        blurred_coarsemap_128 = self.gaussian_blur(coarsemap_128, kernel_size=5, sigma=1.0)
        blurred_gray_128 = self.gaussian_blur(gray_128, kernel_size=5, sigma=1.0)
        
        
        #diff_gray_256 = gray_256 - blurred_gray_256
        #diff_gray_256 = diff_gray_256.repeat(1, 3, 1, 1)
        
        #diff_gray_128 = gray_128 - blurred_gray_128
        #diff_gray_128 = diff_gray_128.repeat(1, 3, 1, 1)
        
        #half_input_map = coarsemap_256 + diff_gray_256
        #half_map_clipped = torch.clamp(half_input_map, 0.0, 1.0)
        #quarter_input_map = coarsemap_128 + diff_gray_128
        #quarter_map_clipped = torch.clamp(quarter_input_map, 0.0, 1.0)
        
    
        origin_map = self.residual6_encoder(coarsemap)
        coarsemap_256 = coarsemap_256.repeat(1, 3, 1, 1)
        half_map = self.residual6_encoder(coarsemap_256)
        coarsemap_128 = coarsemap_128.repeat(1, 3, 1, 1)
        quater_map =self.residual6_encoder(coarsemap_128)

        upsample_4_8 = self.upsampling(quater_map, (8,8))
        half_map = half_map + upsample_4_8
          
        upsample_8_16 = self.upsampling(half_map, (16,16))
        origin_map = origin_map + upsample_8_16
        
        out = self.single_decoder(origin_map)
        
        return out

def get_model(opt, device):
    gen = Generator(opt).to(device)
    ref = Refiner(opt).to(device)
    return gen, ref