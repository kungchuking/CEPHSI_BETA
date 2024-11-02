import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class ste_fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return (x > 0).float()
    @staticmethod
    def backward(ctx, grad):
        return F.hardtanh(grad)

class STE(nn.Module):
    def __init__(self):
        super(STE, self).__init__()
    def forward(self, x):
        return ste_fn.apply(x)

class cep_enc(nn.Module):
    def __init__(self,
            sigma_range=[0, 1e-9],
            ce_code_n=8,
            frame_n=8,
            ce_code_init=None,
            opt_cecode=False,
            patch_size=[720, 1280],
            in_channels=3,
            n_cam=2):
        super(cep_enc, self).__init__()
        self.sigma_range = sigma_range
        self.frame_n = frame_n  # frame num

        self.upsample_factor = frame_n // ce_code_n
        self.ce_code_n = ce_code_n
        
        # -- Added by Chu King on Oct, 29 2024, as these parameters will be used during the invocation of the forward() method.
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.n_cam = n_cam

        # -- Modified by Chu King on Oct, 29, 2024 for pixel-wise CEP
        # -- self.ce_weight = nn.Parameter(torch.Tensor(ce_code_n, 1))
        self.ce_weight = nn.Parameter(torch.Tensor(ce_code_n, in_channels, *patch_size, n_cam))
        
        if ce_code_init is None:
            nn.init.uniform_(self.ce_weight, a=-1, b=1)  # initialize
        else:
            # convert 0,1 (ce_code) -> -1,1 (ce_weight)
            ce_code_init = [-1 if ce_code_init[k] ==
                            0 else 1 for k in range(len(ce_code_init))]
            self.ce_weight.data = torch.tensor(
                ce_code_init, dtype=torch.float32).unsqueeze(0).t()

        if not opt_cecode:
            # whether optimize ce code
            self.ce_weight.requires_grad = False

        self.ste = STE() 
        self.sigmoid = nn.Sigmoid()

        # -- upsample matrix for ce_code(parameters)
        # -- Commented out by Chu King on Oct 29, 2024.
        # -- The upsampling matrix only works when self.ce_weight is a 2D tensor.
        # -- For pixel-wise coded exposure, self.ce_weight will be a 4D tensor (i.e. [ce_code_n, in_channels, H, W])

    def forward(self, frames):
        device = frames.device
        ce_code = self.ste(self.ce_weight)
        # -- ce_code = self.sigmoid(self.ce_weight)

        # -- Added by Chu King on November 1, 2024 for the implementation of the lite version.
        frames = frames[..., :self.patch_size[0], :self.patch_size[1]]

        # -- Modified by Chu King on Oct 29, 2024 for pixel-wise CEP
        ce_code_up = torch.zeros(self.upsample_factor * self.ce_code_n, self.in_channels, *self.patch_size, self.n_cam, device=device)

        # -- Fill the upsampled matrix
        # -- We are not using matrix multiplication for CEP, as the multiplication of two high-dimensional tensors is hard to visualize.
        for i in range(self.ce_code_n):
            ce_code_up[i * self.upsample_factor:(i + 1) * self.upsample_factor, :, :, :, :] = ce_code[i]

        # -- Commented out by Chu King on Oct 29, 2024 as the equation ce_code_up.data.shape[0] == frames.shape[1] no longer holds for pixel-wise CEP.

        # -- Modified by Chu King on Oct 29, 2024 for pixel-wise CEP.
        ce_code_up_ = ce_code_up.repeat(frames.shape[0], 1, 1, 1, 1, 1).to(device)

        ce_blur_img = torch.zeros(frames.shape[0], self.in_channels * 2 * self.n_cam, *self.patch_size, device=device)
        
        for k in range(self.n_cam):
            ce_blur_img[:, (2*k)*self.in_channels:(2*k+1)*self.in_channels, :, :] = torch.sum(ce_code_up_[..., k] * frames, axis=1) / self.frame_n
            ce_blur_img[:, (2*k+1)*self.in_channels:(2*k+2)*self.in_channels, :, :] = torch.sum((1. - ce_code_up_[..., k]) * frames, axis = 1) / self.frame_n

        noise_level = np.random.uniform(*self.sigma_range)

        ce_blur_img_noisy = ce_blur_img + torch.tensor(noise_level, device=device) * torch.randn(ce_blur_img.shape, device=device)

        return ce_blur_img_noisy, ce_code_up_, ce_blur_img

