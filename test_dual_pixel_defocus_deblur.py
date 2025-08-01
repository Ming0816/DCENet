"""
## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881
"""

import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
import numpy as np

from skimage import img_as_ubyte
from basicsr.models.archs.DCENetMultiHead_arch import DCENetMHLocal
import cv2
import utils
from natsort import natsorted
from glob import glob
from pdb import set_trace as stx

import lpips
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
alex = lpips.LPIPS(net='alex').to(device)


parser = argparse.ArgumentParser(description='Dual Pixel Defocus Deblurring using DCE')

parser.add_argument('--input_dir', default='./datasets/Defocus_Deblurring/test/DPDD/', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/Dual_Pixel_Defocus_Deblurring_DCE/', type=str, help='Directory for results')
parser.add_argument('--weights', default='./net_g_DualPixel_DCE.pth', type=str, help='Path to weights')
parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory')

args = parser.parse_args()

####### Load yaml #######
yaml_file = './options/train/Defocus-Deblur/DefocusDeblur_DualPixel_16bit_DCENet.yml'
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)

s = x['network_g'].pop('type')
##########################

model_restoration = DCENetMHLocal(**x['network_g'])

checkpoint = torch.load(args.weights, map_location=device)
model_restoration.load_state_dict(checkpoint['params'])
print("===>Testing using weights: ",args.weights)
model_restoration = model_restoration.to(device)
model_restoration.eval()

result_dir = args.result_dir
if args.save_images:
    os.makedirs(result_dir, exist_ok=True)

filesL = natsorted(glob(os.path.join(args.input_dir, 'inputL', '*.png')))
filesR = natsorted(glob(os.path.join(args.input_dir, 'inputR', '*.png')))
filesC = natsorted(glob(os.path.join(args.input_dir, 'target', '*.png')))

indoor_labels  = np.load('./datasets/Defocus_Deblurring/test/DPDD/indoor_labels.npy')
outdoor_labels = np.load('./datasets/Defocus_Deblurring/test/DPDD/outdoor_labels.npy')

# 預先取得 head 數量，或設你知道的數字
N_heads = 10
psnr = [[] for _ in range(N_heads)]
mae = [[] for _ in range(N_heads)]
ssim = [[] for _ in range(N_heads)]
pips = [[] for _ in range(N_heads)]
with torch.no_grad():
    for fileL, fileR, fileC in tqdm(zip(filesL, filesR, filesC), total=len(filesC)):
        imgL = np.float32(utils.load_img16(fileL)) / 65535.
        imgR = np.float32(utils.load_img16(fileR)) / 65535.
        imgC = np.float32(utils.load_img16(fileC)) / 65535.

        patchL = torch.from_numpy(imgL).unsqueeze(0).permute(0, 3, 1, 2)
        patchR = torch.from_numpy(imgR).unsqueeze(0).permute(0, 3, 1, 2)
        patchC = torch.from_numpy(imgC).unsqueeze(0).permute(0, 3, 1, 2).to(device)
        input_ = torch.cat([patchL, patchR], 1).to(device)
        
        restored = model_restoration(input_)  # [1, N_heads, 6, H, W]

        for h in range(N_heads):
            restored_h = restored[:, h, :, :, :]  # [1, 3, H, W]
            restored_h = torch.clamp(restored_h, 0, 1)

            # LPIPS
            pips[h].append(alex(patchC, restored_h, normalize=True).item())

            # HWC image
            restored_np = restored_h.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

            psnr[h].append(utils.PSNR(imgC, restored_np))
            mae[h].append(utils.MAE(imgC, restored_np))
            ssim[h].append(utils.SSIM(imgC, restored_np))

            if args.save_images:
                save_file = os.path.join(result_dir, f"head{h}_" + os.path.split(fileC)[-1])
                restored_uint16 = np.uint16((restored_np*65535).round())
                utils.save_img(save_file, restored_uint16)
    print("\n=== Per-Head Overall / Indoor / Outdoor Scores ===")
    for h in range(N_heads):
        psnr_0 = np.array(psnr[h])
        mae_0  = np.array(mae[h])
        ssim_0 = np.array(ssim[h])
        pips_0 = np.array(pips[h])

        psnr_indoor  = psnr_0[indoor_labels - 1]
        mae_indoor   = mae_0[indoor_labels - 1]
        ssim_indoor  = ssim_0[indoor_labels - 1]
        pips_indoor  = pips_0[indoor_labels - 1]

        psnr_outdoor = psnr_0[outdoor_labels - 1]
        mae_outdoor  = mae_0[outdoor_labels - 1]
        ssim_outdoor = ssim_0[outdoor_labels - 1]
        pips_outdoor = pips_0[outdoor_labels - 1]

        print("h{:1d} Overall: PSNR {:.4f} SSIM {:.4f} MAE {:.4f} LPIPS {:.4f}".format(h, np.mean(psnr_0), np.mean(ssim_0), np.mean(mae_0), np.mean(pips_0)))
        print("h{:1d} Indoor:  PSNR {:.4f} SSIM {:.4f} MAE {:.4f} LPIPS {:.4f}".format(h, np.mean(psnr_indoor), np.mean(ssim_indoor), np.mean(mae_indoor), np.mean(pips_indoor)))
        print("h{:1d} Outdoor: PSNR {:.4f} SSIM {:.4f} MAE {:.4f} LPIPS {:.4f}".format(h, np.mean(psnr_outdoor), np.mean(ssim_outdoor), np.mean(mae_outdoor), np.mean(pips_outdoor)))

