import torch
from torch.utils.data import Dataset
import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

DEBUG = False

class dataset(Dataset):
    def __init__(self, ds_dir, n_subframe, patch_size=[64, 64]):
        dir_ls = sorted(os.listdir(ds_dir))
        print ("[INFO] Loading dataset from directories: ", dir_ls)
        self.subexp_ls = []
        for d in tqdm(dir_ls):
            f_path = os.path.join(ds_dir, d)
            f_ls = sorted(os.listdir(f_path))
            for i in range(len(f_ls) // n_subframe):
                for j in range(n_subframe):
                    img = cv2.imread(os.path.join(f_path, f_ls[i*n_subframe+j]))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    
                    # -- Separate the whole image into 64x64 tiles
                    R = img.shape[0] // patch_size[0]
                    C = img.shape[1] // patch_size[1]
                    for m in range(R):
                        for n in range(C):
                            if j == 0 and m == 0 and n == 0:
                                # -- Initialize subexp in the first iteration
                                # -- There is a total of n_subframe * R * C images
                                # -- subexp = torch.zeros(n_subframe * R * C, 1, *img.shape)
                                subexp = torch.zeros(R * C, n_subframe, 1, *patch_size)
                            subexp[m*C+n, j, 0, ...] = torch.from_numpy(img[m*patch_size[0]:m*patch_size[0]+patch_size[0], n*patch_size[1]:n*patch_size[1]+patch_size[1]]).float()

                    # -- print ("[INFO] subexp.shape: ", subexp.shape)

                # -- subexp = subexp.unsqueeze(0) # --.unsqueeze(0)
                self.subexp_ls += [subexp]
        self.subexp_ls = torch.cat(self.subexp_ls, dim=0)

        if DEBUG:
            print ("[INFO] subexp_ls.shape: ", self.subexp_ls.shape)
            print (dir_ls)

    def __len__(self):
        return self.subexp_ls.shape[0]
    def __getitem__(self, idx):
        return self.subexp_ls[idx]

if DEBUG:
    train_set = dataset(ds_dir="./dataset/train/",
                        n_subframe=8)
    # test_set = dataset(ds_dir="./dataset/test/",
    #                    n_subframe=8)
    quit()
