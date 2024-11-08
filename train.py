import torch
from dataset import dataset
from cep_system import cep_system
import loss
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from tqdm import tqdm
import os

if __name__ == "__main__":
    # -- Essential directories (Create them if they don't already exist)
    ess_dirs = ["./results/", "./checkpoints/"]
    
    for d in ess_dirs:
        if not os.path.exists(d):
            os.makedirs(d)
    
    fd = open("train.log", "w")
    
    # -- Loss Functions
    TVLoss = loss.TVLoss()
    MSELoss = loss.MSELoss()
    
    # -- Some global parameters
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    
    n_rpt      = 1000       # -- Number of iterations before reporting the loss, PSNR, and other metrics 
    n_epoch    = 5000       # -- Total number of epochs
    n_img      = 100        # -- Number of epochs before saving images for inspection
    patch_size = [512, 512]
    n_subexp   = 16         # -- Number of subexposure per frame
    
    # -- SOme metrics to help evaluate the performance of the model.
    # -- The PSNRs of the last epoch
    train_psnr_o = -1.
    valid_psnr_o = -1.
    # -- The PSNRs of the current epoch
    train_psnr = 0.
    valid_psnr = 0.
    
    # -- The Model
    model = cep_system(sigma_range=[0, 1e-12],
                       ce_code_n=n_subexp,
                       frame_n=n_subexp,
                       opt_cecode=True,
                       n_cam=2,
                       in_channels=1,
                       out_channels=1,
                       patch_size=patch_size).to(device)
    
    params = {
            "batch_size": 8,
            "shuffle": True,
            "num_workers": 1
            }
    
    train_set = dataset(ds_dir="./dataset/train/", n_subframe=n_subexp, patch_size=patch_size)
    train_gen = torch.utils.data.DataLoader(train_set, **params)
    
    print ("[INFO] len(train_set): ", len(train_set))
    print ("[INFO] len(train_set): ", len(train_set), file=fd)
    
    test_set = dataset(ds_dir="./dataset/test/", n_subframe=n_subexp, patch_size=patch_size)
    test_gen = torch.utils.data.DataLoader(test_set, **params)
    
    print ("[INFO] len(test_set): ", len(test_set))
    print ("[INFO] len(test_set): ", len(test_set), file=fd)
    
    # -- optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # for param_tensor in model.state_dict():
    #     print ("{}: {}".format(param_tensor, model.state_dict()[param_tensor].size()))
    # quit()
    
    for epoch in range(n_epoch):
        print ("[INFO] ---- Epoch {} ----".format(epoch), file=fd)
    
        model.train()
    
        # -- Initialize the running sum of some relevant metrics (i.e. Loss, PSNR)
        run_loss = .0
        run_psnr_avg = .0
    
        # -- Training loop
        for i, data in tqdm(enumerate(train_gen)):
            data = data.to(device).float() / 255.
    
            # -- Zero the gradients for each data
            optimizer.zero_grad()
    
            # -- Make predictions
            # -- target == data (i.e. autoencoder)
            output, ce_blur, _, target, reblur = model(data)
    
            # -- Compute the loss
            output_ = torch.flatten(output, end_dim=1)
            target_ = torch.flatten(target, end_dim=1)
            ce_blur_ = torch.flatten(ce_blur, end_dim=1)
            reblur_ = torch.flatten(reblur, end_dim=1)
    
            loss_ml = MSELoss(output_, target_)
            loss_reblur = 0.2 * MSELoss(ce_blur_, reblur_)
            loss_tv = 0.2 * TVLoss(output)
            loss = loss_ml + loss_reblur + loss_tv
    
            # print ("[INFO] loss_ml: ", loss_ml)
            # print ("[INFO] loss_reblur: ", loss_reblur)
            # print ("[INFO] loss_tv: ", loss_tv)
    
            # -- Clamp the output prior to computing the PSNR
            output = torch.clamp(output, min=0., max=1.)
    
            # -- Iterate over each batch and each subframe to calculate the average PSNR of the current batch.
            run_psnr = .0
            for m in range(output.shape[0]):
                for n in range(output.shape[1]):
                    run_psnr += psnr(output[m, n, 0, ...].detach().cpu().numpy(), target[m, n, 0, ...].detach().cpu().numpy())
            psnr_avg = run_psnr / float(output.shape[0] * output.shape[1])
    
            # -- Compute gradients
            loss.backward()
    
            # -- Adjust the weights
            optimizer.step()
    
            run_loss += loss.item()
            run_psnr_avg += psnr_avg
            # -- Report the performance once every n_rpt iterations
            if i % n_rpt == 0 and i != 0:
                print ("[INFO] Iteration {}; Loss: {}; PSNR: {};".format(i, run_loss / float(n_rpt), run_psnr_avg / float(i+1)), file=fd)
                run_loss = .0
    
        train_psnr = run_psnr_avg / float(i+1)
        print ("[INFO] Training Set PSNR: {};".format(train_psnr), file=fd)
    
        # -- Save some output images for inspectin once every n_img iterations
        if epoch % n_img == 0 and epoch != 0:
            for n in range(output.shape[1]):
                plt.imshow(output[0, n, 0, ...].detach().cpu().numpy(), cmap="gray")
                plt.savefig("./results/train_epoch_{}_n_{}.png".format(epoch, n))
    
        """
        print ("[INFO] ce_blur.shape: ", ce_blur.shape)
        print ("[INFO] output.shape: ", output.shape)
        print ("[INFO] target.shape: ", output.shape)
    
        print ("[INFO] ce_blur_.shape: ", ce_blur_.shape)
        print ("[INFO] output_.shape: ", output_.shape)
        print ("[INFO] target_.shape: ", output_.shape)
        """
    
        # -- Validation
        # -- Initialize the running sum of some relevant metrics (i.e. PSNR)
        run_psnr_avg = .0
        with torch.set_grad_enabled(False):
            for i, data in tqdm(enumerate(test_gen)):
                data = data.to(device).float() / 255.
    
                # -- Make predictions
                # -- target == data (i.e. autoencoder)
                output, ce_blur, _, target, reblur = model(data)
    
                # -- Clamp the output prior to computing the PSNR
                output = torch.clamp(output, min=0., max=1.)
    
                # -- Iterate over each batch and each subframe to calculate the average PSNR of the current batch.
                run_psnr = .0
                for m in range(output.shape[0]):
                    for n in range(output.shape[1]):
                        run_psnr += psnr(output[m, n, 0, ...].detach().cpu().numpy(), target[m, n, 0, ...].detach().cpu().numpy())
                psnr_avg = run_psnr / float(output.shape[0] * output.shape[1])
    
                run_psnr_avg += psnr_avg
    
            # -- Report the performance on the validation set
            valid_psnr = run_psnr_avg / float(i+1)
            print ("[INFO] Validation Set PSNR: {};".format(valid_psnr), file=fd)
    
        fd.flush()
    
        # -- Save some output images for inspection once every n_img iterations
        if epoch % n_img == 0 and epoch != 0:
            for n in range(output.shape[1]):
                plt.imshow(output[0, n, 0, ...].detach().cpu().numpy(), cmap="gray")
                plt.savefig("./results/valid_epoch_{}_n_{}.png".format(epoch, n))
    
        # -- Save the parameters as checkpoints if they yield higher PSNRs than those of previous epochs.
        if train_psnr > train_psnr_o and valid_psnr > valid_psnr_o:
            train_psnr_o = train_psnr
            valid_psnr_o = valid_psnr
            torch.save(model.state_dict(), "./checkpoints/epoch_{}_valid_psnr_{}_train_psnr_{}.pth".format(epoch, valid_psnr, train_psnr))
