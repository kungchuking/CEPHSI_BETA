import torch.nn as nn
from cep_enc import cep_enc

# -- Added by Chu King on Oct 28, 2024 for the deployment of MobileNetV2 Auto Encoder
from autoencoder import autoencoder

class cep_system(nn.Module):
    def __init__(self,
            sigma_range=0,
            ce_code_n=8,
            frame_n=8,
            ce_code_init=None,
            opt_cecode=False,
            in_channels=3,
            n_cam=2,
            patch_size=[720, 1280],
            out_channels=3):
        super(cep_system, self).__init__()
        self.ce_code_n = ce_code_n
        self.frame_n = frame_n
        self.patch_size = patch_size

        self.cep_enc = cep_enc(
            sigma_range=sigma_range,
            ce_code_n=ce_code_n,
            frame_n=frame_n,
            ce_code_init=ce_code_init,
            opt_cecode=opt_cecode,
            patch_size=patch_size,
            # -- Added by Chu King on OCT 29, 2024 for grayscale images and pixelwise CEP.
            in_channels=in_channels,
            n_cam=n_cam)

        self.cep_dec = autoencoder(
            in_channels=in_channels,
            out_channels=out_channels,
            frame_n = self.frame_n,
            n_feats=4,
            n_cam=n_cam)

    def forward(self, frames):
        ce_blur_img_noisy, ce_code_up, ce_blur_img = self.cep_enc(frames)

        output = self.cep_dec(ce_blur=ce_blur_img_noisy, ce_code=ce_code_up)

        _, _, reblur = self.cep_enc(output)

        return output, ce_blur_img, ce_blur_img_noisy, frames[..., :self.patch_size[0], :self.patch_size[1]], reblur
