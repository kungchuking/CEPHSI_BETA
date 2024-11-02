import torch
import torch.nn as nn

class autoencoder(nn.Module):
    def __init__(self,
            in_channels=3,
            out_channels=3,
            frame_n=8,
            n_feats=8,
            n_cam=2):
        super(autoencoder, self).__init__()
        self.frame_n = frame_n
        self.in_channels = in_channels
        # -- Mask Encoder
        self.mask_enc1 = nn.Sequential(
            nn.Conv2d(in_channels * 1 * n_cam * frame_n, 8 * n_feats, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8 * n_feats),
            nn.ReLU())
        self.mask_enc2 = nn.Sequential(
            nn.Conv2d(8 * n_feats, 8 * n_feats, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8 * n_feats),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # -- Image Encoder
        self.img_enc1 = nn.Sequential(
            nn.Conv2d(in_channels * 2 * n_cam, 8 * n_feats, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8 * n_feats),
            nn.ReLU())
        self.img_enc2 = nn.Sequential(
            nn.Conv2d(8 * n_feats, 8 * n_feats, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8 * n_feats),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # -- Aggregate Encoder
        self.enc3 = nn.Sequential(
            nn.Conv2d(16 * n_feats, 16 * n_feats, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16 * n_feats),
            nn.ReLU())
        self.enc4 = nn.Sequential(
            nn.Conv2d(16 * n_feats, 16 * n_feats, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16 * n_feats),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.enc5 = nn.Sequential(
            nn.Conv2d(16 * n_feats, 32 * n_feats, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32 * n_feats),
            nn.ReLU())
        self.enc6 = nn.Sequential(
            nn.Conv2d(32 * n_feats, 32 * n_feats, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32 * n_feats),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # -- Decoder
        self.dec1 = nn.Sequential(
            nn.Conv2d(32 * n_feats, 32* n_feats, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32 * n_feats),
            nn.ReLU())
        self.dec2 = nn.Sequential(
            nn.Conv2d(32 * n_feats, 16 * n_feats, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16 * n_feats),
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2))
        self.dec3 = nn.Sequential(
            nn.Conv2d(16 * n_feats, 16 * n_feats, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16 * n_feats),
            nn.ReLU())
        self.dec4 = nn.Sequential(
            nn.Conv2d(16 * n_feats, 8 * n_feats, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8 * n_feats),
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2))
        self.dec5 = nn.Sequential(
            nn.Conv2d(8 * n_feats, 8 * n_feats, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8 * n_feats),
            nn.ReLU())
        self.dec6 = nn.Sequential(
            nn.Conv2d(8 * n_feats, out_channels * frame_n, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels * frame_n),
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2))
        self.sigmoid = nn.Sequential(
            nn.Conv2d(out_channels * frame_n, out_channels * frame_n, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid())
    def forward(self, ce_blur, ce_code):

        # -- print ("[INFO] ce_blur.shape: ", ce_blur.shape)
        # -- print ("[INFO] ce_code.shape: ", ce_code.shape)

        ce_code_resize = torch.zeros(ce_code.shape[0], ce_code.shape[1] * ce_code.shape[2] * ce_code.shape[5], *ce_code.shape[3:5])
        for i in range(ce_code.shape[1]):
            for j in range(ce_code.shape[2]):
                for k in range(ce_code.shape[5]):
                    ce_code_resize[:, i * ce_code.shape[2] * ce_code.shape[5] + j * ce_code.shape[5] + k, ...] = ce_code[:, i, j, :, :, k]
        # -- Mask Encoder
        y = self.mask_enc1(ce_code_resize)
        y = self.mask_enc2(y)

        # -- Image Encoder
        x = self.img_enc1(ce_blur)
        x = self.img_enc2(x)

        # -- print ("y.shape: ", y.shape)
        # -- print ("x.shape: ", x.shape)
        # -- quit()

        z = torch.zeros(x.shape[0], x.shape[1] * 2, x.shape[2], x.shape[3])
        z[:, :x.shape[1], ...] = x
        z[:, x.shape[1]:, ...] = y

        # -- Merging ce_blur and ce_code
        x = self.enc3(z)
        x = self.enc4(x)
        x = self.enc5(x)
        x = self.enc6(x)
        # -- Decoder
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)
        x = self.dec5(x)
        x = self.dec6(x)
        # -- Sigmoid
        x = self.sigmoid(x)
        return torch.reshape(x, (-1, self.frame_n, self.in_channels, *ce_blur.shape[-2:]))
