import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import numpy as np
import pandas as pd
import pickle

import cv2
from PIL import Image
from torchvision import transforms
import torch
import random

def reseed(seed):
    random.seed(seed)
    torch.manual_seed(seed)

def load_image_in_PIL_to_Tensor(path, mode='RGB', transform=None):
    img_PIL = Image.open(path).convert(mode)
    if transform:
        img_tensor = transform(img_PIL)
        return img_tensor
    return img_PIL


def load_audio_lm(audio_lm_path):
    with open(audio_lm_path, 'rb') as fr:
        audio_log_mel = pickle.load(fr)
    audio_log_mel = audio_log_mel.detach()  # [5, 1, 96, 64]
    return audio_log_mel


class AVAffordanceDataset(Dataset):
    """Dataset for multiple sound source segmentation"""

    def __init__(self, split='train', cfg=None):
        super(AVAffordanceDataset, self).__init__()
        self.split = split
        self.mask_num = 1
        self.cfg = cfg
        df_all = pd.read_csv(cfg.anno_csv, sep=',')
        self.df_split = df_all
        im_mean = (124, 116, 104)
        print("{}/{} videos are used for {}".format(len(self.df_split),
              len(df_all), self.split))
        self.img_transform_train = transforms.Compose([
            transforms.RandomAffine(degrees=20, scale=(0.5,1.0), shear=10),
            transforms.Resize([512, 512], Image.BILINEAR),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.mask_transform_train = transforms.Compose([
            transforms.RandomAffine(degrees=20, scale=(0.5,1.0), shear=10),
            transforms.Resize([512, 512], Image.NEAREST),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        self.img_transform = transforms.Compose([
            transforms.Resize([512, 512], Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize([512, 512], Image.NEAREST),
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        sequence_seed = np.random.randint(2147483647)
        df_one_video = self.df_split.iloc[index]
        img_base_path = os.path.join(self.cfg.dir_img, df_one_video["image"])
        audio_lm_path = os.path.join(
            self.cfg.dir_audio_log_mel, df_one_video["pkl"])
        mask_func_base_path = os.path.join(
            self.cfg.dir_mask, df_one_video["mask_func"])
        mask_dep_base_path = os.path.join(
            self.cfg.dir_mask, df_one_video["mask_dep"])
        audio_log_mel = load_audio_lm(audio_lm_path)
        # audio_lm_tensor = torch.from_numpy(audio_log_mel)
        imgs, masks_func,masks_dep = [], [],[]
        if self.split == "train":
            reseed(sequence_seed)
            img = load_image_in_PIL_to_Tensor(img_base_path, transform=self.img_transform_train)
            imgs.append(img)
            reseed(sequence_seed)
            mask_func = load_image_in_PIL_to_Tensor(mask_func_base_path, transform=self.mask_transform_train, mode='P')
            masks_func.append(mask_func)
            reseed(sequence_seed)
            mask_dep = load_image_in_PIL_to_Tensor(mask_dep_base_path, transform=self.mask_transform_train, mode='P')
            masks_dep.append(mask_dep)
        else:
            img = load_image_in_PIL_to_Tensor(img_base_path, transform=self.img_transform)
            imgs.append(img)
            mask_func = load_image_in_PIL_to_Tensor(mask_func_base_path, transform=self.mask_transform, mode='P')
            masks_func.append(mask_func)
            mask_dep = load_image_in_PIL_to_Tensor(mask_dep_base_path, transform=self.mask_transform, mode='P')
            masks_dep.append(mask_dep)
        imgs_tensor = torch.stack(imgs, dim=0)
        masks_func_tensor = torch.stack(masks_func, dim=0)
        masks_dep_tensor = torch.stack(masks_dep, dim=0)
        class_label = torch.tensor([0, 1]).view(1,-1)
        return imgs_tensor, audio_log_mel, masks_func_tensor,masks_dep_tensor,class_label, df_one_video["image"].replace(".jpg","").replace("/","_")+"_"+df_one_video["audio"].replace(".mp3","").split("/")[-1]

    def __len__(self):
        # return 2025
        return len(self.df_split)
