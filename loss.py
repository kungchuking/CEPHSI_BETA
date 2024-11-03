import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pypher.pypher import psf2otf
from tqdm import tqdm

class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, output, target):
        return F.mse_loss(output, target)

class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, output, *args):
        # check if GPU is available, otherwise use CPU
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # finite differences kernels and corresponding otfs
        dx = np.array([[-1., 1.]])
        dy = np.array([[-1.], [1.]])
        dxFT = torch.from_numpy(psf2otf(dx, output.shape[-2:])).to(device)
        dyFT = torch.from_numpy(psf2otf(dy, output.shape[-2:])).to(device)
        dxyFT = torch.stack((dxFT, dyFT), axis=0)

        grad_fn = lambda x: torch.stack([torch.fft.ifft2(torch.fft.fft2(x) * dxyFT[i, :, :]) for i in range(dxyFT.shape[0])], dim=0)

        run_loss = .0
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                for k in range(output.shape[2]):
                    run_loss += torch.sum(torch.norm(grad_fn(output[i, j, k, ...]), dim=0))

        denom = float(output.shape[0] * output.shape[1] * output.shape[2] * output.shape[3] * output.shape[4])
        return run_loss / denom
