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
        # -- 512 x 512
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
        # -- 512 x 512
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
        # -- 256 x 256
        self.enc3 = nn.Sequential(
            nn.Conv2d(16 * n_feats, 16 * n_feats, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16 * n_feats),
            nn.ReLU())
        self.enc4 = nn.Sequential(
            nn.Conv2d(16 * n_feats, 16 * n_feats, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16 * n_feats),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # -- 128 x 128
        self.enc5 = nn.Sequential(
            nn.Conv2d(16 * n_feats, 32 * n_feats, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32 * n_feats),
            nn.ReLU())
        self.enc6 = nn.Sequential(
            nn.Conv2d(32 * n_feats, 32 * n_feats, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32 * n_feats),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # -- 64 x 64
        self.enc7 = nn.Sequential(
            nn.Conv2d(32 * n_feats, 64 * n_feats, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64 * n_feats),
            nn.ReLU())
        self.enc8 = nn.Sequential(
            nn.Conv2d(64 * n_feats, 64 * n_feats, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64 * n_feats),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # -- Decoder
        # -- 32 x 32
        self.dec8 = nn.Sequential(
            nn.Conv2d(64 * n_feats, 64* n_feats, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64 * n_feats),
            nn.ReLU())
        self.dec7 = nn.Sequential(
            nn.Conv2d(64 * n_feats, 32 * n_feats, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32 * n_feats),
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2))
        # -- 64 x 64 
        self.dec6 = nn.Sequential(
            nn.Conv2d(32 * n_feats, 32* n_feats, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32 * n_feats),
            nn.ReLU())
        self.dec5 = nn.Sequential(
            nn.Conv2d(32 * n_feats, 16 * n_feats, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16 * n_feats),
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2))
        # -- 128 x 128
        self.dec4 = nn.Sequential(
            nn.Conv2d(16 * n_feats, 16 * n_feats, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16 * n_feats),
            nn.ReLU())
        self.dec3 = nn.Sequential(
            nn.Conv2d(16 * n_feats, 16 * n_feats, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16 * n_feats),
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2))
        # -- 256 x 256
        self.dec2 = nn.Sequential(
            nn.Conv2d(16 * n_feats, 8 * n_feats, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8 * n_feats),
            nn.ReLU())
        self.dec1 = nn.Sequential(
            nn.Conv2d(8 * n_feats, out_channels * frame_n, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels * frame_n),
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2))
        # -- 512 x 512
        self.sigmoid = nn.Sequential(
            nn.Conv2d(out_channels * frame_n, out_channels * frame_n, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid())
    def forward(self, ce_blur, ce_code):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        ce_blur = ce_blur.to(device)
        ce_code = ce_code.to(device)
        
        # -- print ("[INFO] ce_blur.shape: ", ce_blur.shape)
        # -- print ("[INFO] ce_code.shape: ", ce_code.shape)

        ce_code_resize = torch.zeros(ce_code.shape[0], ce_code.shape[1] * ce_code.shape[2] * ce_code.shape[5], *ce_code.shape[3:5])
        for i in range(ce_code.shape[1]):
            for j in range(ce_code.shape[2]):
                for k in range(ce_code.shape[5]):
                    ce_code_resize[:, i * ce_code.shape[2] * ce_code.shape[5] + j * ce_code.shape[5] + k, ...] = ce_code[:, i, j, :, :, k]

        ce_code_resize = ce_code_resize.to(device)
        # -- Mask Encoder
        y = self.mask_enc1(ce_code_resize)
        y = self.mask_enc2(y)

        # -- Image Encoder
        x = self.img_enc1(ce_blur)
        x = self.img_enc2(x)

        # -- print ("y.shape: ", y.shape)
        # -- print ("x.shape: ", x.shape)
        # -- quit()

        z = torch.zeros(x.shape[0], x.shape[1] * 2, x.shape[2], x.shape[3]).to(device)
        z[:, :x.shape[1], ...] = x
        z[:, x.shape[1]:, ...] = y

        # -- Merging ce_blur and ce_code
        x1 = self.enc3(z)
        x2 = self.enc4(x1)
        x3 = self.enc5(x2)
        x4 = self.enc6(x3)
        x5 = self.enc7(x4)
        x6 = self.enc8(x5)
        # -- Decoder
        x7 = self.dec8(x6)
        x8 = self.dec7(x7)
        x9 = self.dec6(x8 + x4) # -- 64 x 64
        xa = self.dec5(x9)
        xb = self.dec4(xa + x2) # -- 128 x 128
        xc = self.dec3(xb)
        xd = self.dec2(xc + z)  # -- 256 x 256 
        xe = self.dec1(xd)
        # -- Sigmoid
        xf = self.sigmoid(xe)
        return torch.reshape(xf, (-1, self.frame_n, self.in_channels, *ce_blur.shape[-2:]))
