import torch
from dataset import dataset
from cebd_model import CEBDNet
import _pix_loss_cls
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr

# -- Loss Functions
CharbonnierLoss = _pix_loss_cls.CharbonnierLoss()
SSIMLoss = _pix_loss_cls.SSIMLoss(channel=1)
EdgeLoss = _pix_loss_cls.EdgeLoss(channel=1)

WeightedLoss = lambda x, y : CharbonnierLoss(x, y) + 0.05 * SSIMLoss(x, y) + 0.05 * EdgeLoss(x, y)

# -- Some global parameters
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

n_rpt   = 5   # -- Number of iterations before reporting the loss, PSNR, and other metrics 
n_epoch = 500 # -- Total number of epochs
n_img   = 10  # -- Number of epochs before saving images for inspection

# -- The Model
model = CEBDNet(sigma_range=[0, 1e-12],
                ce_code_n=8,
                frame_n=8,
                opt_cecode=True,
                ce_net="CEBlurNet",
                binary_fc="STEBinary_fc",
                bd_net="MobileNetV2CAE",
                n_cam=2,
                in_channels=1,
                out_channels=1,
                patch_size=[512, 512])

params = {
        "batch_size": 8,
        "shuffle": True,
        "num_workers": 1
        }

train_set = dataset(ds_dir="./dataset/train/", n_subframe=8)
train_gen = torch.utils.data.DataLoader(train_set, **params)

print ("[INFO] len(train_set): ", len(train_set))

test_set = dataset(ds_dir="./dataset/test/", n_subframe=8)
test_gen = torch.utils.data.DataLoader(test_set, **params)

print ("[INFO] len(test_set): ", len(test_set))

# -- optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

for epoch in range(n_epoch):
    print ("[INFO] ---- Epoch {} ----".format(epoch))

    model.train()

    # -- Initialize the running sum of some relevant metrics (i.e. Loss, PSNR)
    run_loss = .0
    run_psnr_avg = .0

    # -- Training loop
    for i, data in enumerate(train_gen):
        data = data.to(device).float() / 255.

        # -- Zero the gradients for each data
        optimizer.zero_grad()

        # -- Make predictions
        # -- target == data (i.e. autoencoder)
        output, ce_blur, _, target, reblur = model(data)

        # -- Iterate over each batch and each subframe to calculate the average PSNR of the current batch.
        run_psnr = .0
        for m in range(output.shape[0]):
            for n in range(output.shape[1]):
                run_psnr += psnr(output[m, n, 0, ...].detach().numpy(), target[m, n, 0, ...].detach().numpy())
        psnr_avg = run_psnr / float(output.shape[0] * output.shape[1])

        # -- Compute the loss
        output_ = torch.flatten(output, end_dim=1)
        target_ = torch.flatten(target, end_dim=1)
        ce_blur_ = torch.flatten(ce_blur, end_dim=1)
        reblur_ = torch.flatten(reblur, end_dim=1)

        loss = WeightedLoss(output_, target_) + 0.2 * CharbonnierLoss(ce_blur_, reblur_) 

        # -- Compute gradients
        loss.backward()

        # -- Adjust the weights
        optimizer.step()

        run_loss += loss.item()
        run_psnr_avg += psnr_avg
        # -- Report the performance once every n_rpt iterations
        if i % n_rpt == 0 and i != 0:
            print ("[INFO] Iteration {}; Loss: {}; PSNR: {};".format(i, run_loss / float(n_rpt), run_psnr_avg / float(n_rpt)))
            run_loss = .0
            run_psnr_avg = .0

    # -- Save some output images for inspectin once every n_img iterations
    if epoch % n_img == 0 and epoch != 0:
        for n in range(output.shape[1]):
            plt.imshow(output[0, n, 0, ...].detach().numpy(), cmap="gray")
            plt.savefig("./results/epoch_{}_n_{}.png".format(epoch, n))

    print ("[INFO] ce_blur.shape: ", ce_blur.shape)
    print ("[INFO] output.shape: ", output.shape)
    print ("[INFO] target.shape: ", output.shape)

    print ("[INFO] ce_blur_.shape: ", ce_blur_.shape)
    print ("[INFO] output_.shape: ", output_.shape)
    print ("[INFO] target_.shape: ", output_.shape)

    # -- Validation
    # -- Initialize the running sum of some relevant metrics (i.e. PSNR)
    run_psnr_avg = .0
    with torch.set_grad_enabled(False):
        for data in test_gen:
            data = data.to(device).float() / 255.

            # -- Make predictions
            # -- target == data (i.e. autoencoder)
            output, ce_blur, _, target, reblur = model(data)

            # -- Iterate over each batch and each subframe to calculate the average PSNR of the current batch.
            run_psnr = .0
            for m in range(output.shape[0]):
                for n in range(output.shape[1]):
                    run_psnr += psnr(output[m, n, 0, ...].detach().numpy(), target[m, n, 0, ...].detach().numpy())
            psnr_avg = run_psnr / float(output.shape[0] * output.shape[1])

            run_psnr_avg += psnr_avg

        # -- Report the performance on the validation set
        print ("[INFO] Validation Set PSNR: {};".format(run_psnr_avg / float(len(test_set))))
        run_psnr_avg = .0

