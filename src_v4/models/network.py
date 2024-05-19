# ------------------------------------------------------------------------
# Copyright (c) 2023 CandleLabAI. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Network definition for Livenet
"""

# imports
from .blocks import GrayEncoder, ResidualEncoder, ResidualEncoder_6, ResidualEncoder_5, ResidualEncoder_4, DecoderBlock_S, BasicBlock, SingleDecoder, ResidualDecoder, RefineBlock, DecoderBlock
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
        self.residual5_encoder = ResidualEncoder_5(in_nc=opt["tranmission_map"]["in_nc"])
        self.residual4_encoder = ResidualEncoder_4(in_nc=opt["tranmission_map"]["in_nc"])
        self.single_decoder = SingleDecoder(out_nc=opt["tranmission_map"]["out_nc"])
        
        self.conv_1x1 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0)
        self.conv_1x1_ch = nn.Conv2d(256*3, 256, kernel_size=1, stride=1, padding=0)
        self.resize_256 = Resize((256, 256))
        self.resize_128 = Resize((128,128))
        
    def forward(self, coarsemap, gray):
        gray = gray.repeat(1, 3, 1, 1)

        half_image = coarsemap[:, :, 1::2, 1::2]
        quater_image = coarsemap[:, :, 1::4, 1::4]
        resized_256 = self.resize_256(coarsemap)
        resized_128 = self.resize_128(coarsemap)

        half_diff_image = resized_256 - half_image
        quater_diff_image = resized_128 - quater_image
    
        origin_map = self.residual6_encoder(coarsemap)
        half_map = self.residual5_encoder(half_diff_image)
        quater_map =self.residual4_encoder(quater_diff_image)
        quater_map = self.conv_1x1(quater_map)
        
        multi_map = torch.cat([origin_map, half_map, quater_map], dim=1)
        multi_map = self.conv_1x1_ch(multi_map)
        out = self.single_decoder(multi_map)
        
        return out

def get_model(opt, device):
    gen = Generator(opt).to(device)
    ref = Refiner(opt).to(device)
    return gen, ref