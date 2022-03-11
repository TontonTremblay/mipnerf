import subprocess
import argparse
import os
import numpy as np 
import glob

from common import ROOT_DIR
from common import compute_error,read_image,linear_to_srgb,downsample

parser = argparse.ArgumentParser(description="Compare group of runs.")

parser.add_argument("--gu_folder", default="", help="guess folder")
parser.add_argument("--gt_folder", default="", help="gt folder")
parser.add_argument("--res", default=400, type=int, help="gt folder")
parser.add_argument("--out", default="exp.csv", help="out_file")
parser.add_argument("--set", default="test", help="out_file")

opt = parser.parse_args()

scene_name = opt.gt_folder.split("/")[-2]

frame_error = {"scene":scene_name,'file_name':"","ssim":0,"psnr":0,"lpips":0,"depth_rmse":0,"depth_mae":0}
all_error = []

import lpips
import torch 
import torchvision.transforms as T
import cv2
import matplotlib.pyplot as plt

transform = T.Compose([
    # T.ToPILImage(),
    # T.Resize(300),
    T.ToTensor()]
)
loss_fn_alex = lpips.LPIPS(net='alex')
loss_fn_alex.eval()

gu_images = sorted(glob.glob(opt.gu_folder+"/*.png"))
gu_depths = sorted(glob.glob(opt.gu_folder+"/*.npy"))

gt_images = []
gt_depths = []

for img_path in sorted(glob.glob(opt.gt_folder+"/*.exr")):
    if "seg" in img_path:
        continue

    number = int(img_path.split("/")[-1].split('.')[0])

    if opt.set == 'test':
        if number < 105:
            continue

    if "depth" in img_path:
        gt_depths.append(img_path)
    else:
        gt_images.append(img_path)



for i_image in range(len(gt_images)):

    # image_srgb = read_image(gu_images[i_image])
    image_srgb = (cv2.imread(gu_images[i_image])/255.0).astype(np.float32)
    # image_srgb = linear_to_srgb(image)

    ref_img_srgb = read_image(gt_images[i_image])
    ref_img_srgb = linear_to_srgb(ref_img_srgb)

    # resizing 
    ref_img_srgb = cv2.resize(ref_img_srgb*255,(opt.res,opt.res))/255.0
    image_srgb = cv2.resize(image_srgb*255,(opt.res,opt.res))/255.0

    # image_srgb[...,:3] += (1.0 - image_srgb[...,3:4]) * 1.0            
    ref_img_srgb[...,:3] += (1.0 - ref_img_srgb[...,3:4]) * 1.0
    ref_img_srgb= ref_img_srgb[:,:,:3]
    ref_img_srgb = cv2.cvtColor(ref_img_srgb, cv2.COLOR_BGR2RGB)
    ref_img_srgb[ref_img_srgb>1]=1

    ssim = compute_error("SSIM",image_srgb,ref_img_srgb)
    psnr = compute_error("MSE",image_srgb,ref_img_srgb)
    print(-10 * np.log(psnr)/np.log(10.))
    frame_error['psnr'] = -10 * np.log(psnr)/np.log(10.)
    frame_error['ssim'] = ssim
    
    # raise()

    # lpips section
    image_srgb_torch = transform(image_srgb).unsqueeze(0)[:,:3,:,:]
    ref_srgb_torch = transform(ref_img_srgb).unsqueeze(0)[:,:3,:,:]
    lpips_v = loss_fn_alex(image_srgb_torch,ref_srgb_torch).item()
    frame_error['lpips'] = lpips_v

    with open(gu_depths[i_image], 'rb') as f:
        depth = np.load(f)

    ref_depth = cv2.imread(gt_depths[i_image],  
                            cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) 
    ref_depth[ref_depth>3.4028235e+37] = 0
    ref_depth[ref_depth<-3.4028235e+37] = 0                         
    ref_depth = ref_depth[:,:,0]

    # turn into a depth image
    # "cx": 800.0,
    # "cy": 800.0,
    # "fx": 1931.371337890625,
    # "fy": 1931.371337890625
    def convert_from_uvd(u, v, d,fx=1931.371337890625,fy=1931.371337890625,cx=800,cy=800):
        # d *= self.pxToMetre
        x_over_z = (cx - u) / fx
        y_over_z = (cy - v) / fy
        z = d / np.sqrt(1. + x_over_z**2 + y_over_z**2)
        x = x_over_z * z
        y = y_over_z * z
        return z

    # for i in range(1600):
    #     for j in range(1600):
    #         ref_depth[i,j] = convert_from_uvd(i,j,ref_depth[i,j])



    # compute metric for depth 

    depth = cv2.resize(depth,(400,400),cv2.INTER_NEAREST)
    ref_depth = cv2.resize(ref_depth,(400,400),cv2.INTER_NEAREST)


    depth_mae = compute_error("MAE",depth,ref_depth,greater_zero=True)
    depth_rmse = compute_error("MRSE",depth,ref_depth,greater_zero=True)
    frame_error['depth_rmse'] = depth_rmse
    frame_error['depth_mae'] = depth_mae
    # print(depth_rmse,depth_mae)
    # raise()

    def make_depth_image(depth,name,max_depth=None):
        # print(max_depth)
        if max_depth is None:
            depth /= depth.max()
        else:
            depth /= max_depth
        depth *= 255.0
        depth = depth.astype(np.uint8)
        cv2.imwrite(name,depth)

    # if i_image <4:
    #     make_depth_image(ref_depth.copy(),f"depth_gt.png",max(ref_depth.max(),depth.max()))
    #     make_depth_image(depth.copy(),f"depth_gu.png",max(ref_depth.max(),depth.max()))
    #     raise()
    # else:
    #     raise()
    all_error.append(frame_error.copy())

import pandas as pd 
df = pd.DataFrame(all_error)

df.to_csv(f"{opt.out}")

print("lpips",df['lpips'].mean())
print("psnr",df['psnr'].mean())
print("ssim",df['ssim'].mean())
print("depth_rmse",df['depth_rmse'].mean())
print("depth_mae",df['depth_mae'].mean())

# subprocess.call(['rm', '-rf', path_scene])