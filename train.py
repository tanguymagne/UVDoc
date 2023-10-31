import argparse
import gc
import os

import torch

import data_doc3D
import data_UVDoc
import model
import utils
from data_mixDataset import mixDataset

train_mse = 0.0
losscount = 0
gamma_w = 0.0


def setup_data(args):
    """
    Returns train and validation dataloader.
    """
    doc3D = data_doc3D.doc3DDataset
    UVDoc = data_UVDoc.UVDocDataset
    traindata = "train"
    valdata = "val"

    # Training data
    t_doc3D_data = doc3D(
        data_path=args.data_path_doc3D,
        split=traindata,
        appearance_augmentation=args.appearance_augmentation,
    )
    t_UVDoc_data = UVDoc(
        data_path=args.data_path_UVDoc,
        appearance_augmentation=args.appearance_augmentation,
        geometric_augmentations=args.geometric_augmentationsUVDoc,
    )
    t_mix_data = mixDataset(t_doc3D_data, t_UVDoc_data)
    if args.data_to_use == "both":
        trainloader = torch.utils.data.DataLoader(
            t_mix_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True
        )
    elif args.data_to_use == "doc3d":
        trainloader = torch.utils.data.DataLoader(
            t_doc3D_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True
        )
    else:
        raise ValueError(f"data_to_use should be either doc3d or both, provided {args.data_to_use}.")

    # Validation data (doc3D only)
    v_doc3D_data = doc3D(data_path=args.data_path_doc3D, split=valdata, appearance_augmentation=[])
    valloader = torch.utils.data.DataLoader(
        v_doc3D_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True
    )

    return trainloader, valloader


def get_scheduler(optimizer, args, epoch_start):
    """Return a learning rate scheduler
    Parameters:
        optimizer          -- the optimizer of the network
        args               -- stores all the experiment flags
        epoch_start        -- the epoch number we started/continued from
    We keep the same learning rate for the first <args.n_epochs> epochs
    and linearly decay the rate to zero over the next <args.n_epochs_decay> epochs.
    """

    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch + epoch_start - args.n_epochs) / float(args.n_epochs_decay + 1)
        return lr_l

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

    return scheduler


def update_learning_rate(scheduler, optimizer):
    """Update learning rates; called at the end of every epoch"""
    old_lr = optimizer.param_groups[0]["lr"]
    scheduler.step()
    lr = optimizer.param_groups[0]["lr"]
    print("learning rate update from %.7f -> %.7f" % (old_lr, lr))
    return lr


def write_log_file(log_file_name, loss, epoch, lrate, phase):
    with open(log_file_name, "a") as f:
        f.write("\n{} LRate: {} Epoch: {} MSE: {:.5f} ".format(phase, lrate, epoch, loss))


def main_worker(args):
    # setup training data
    trainloader, valloader = setup_data(args)

    device = torch.device("cuda:0")
    UVDocnet = model.UVDocnet(num_filter=32, kernel_size=5)
    UVDocnet.to(device)

    # define loss functions
    criterionL1 = torch.nn.L1Loss()
    criterionMSE = torch.nn.MSELoss()

    # initialize optimizers
    optimizer = torch.optim.Adam(UVDocnet.parameters(), lr=args.lr, betas=(0.9, 0.999))

    global gamma_w
    epoch_start = 0

    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("Loading model and optimizer from checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)

            UVDocnet.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            print("Loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint["epoch"]))
            epoch_start = checkpoint["epoch"]
            if epoch_start >= args.ep_gamma_start:
                gamma_w = args.gamma_w
        else:
            print("No checkpoint found at '{}'".format(args.resume))

    # initialize learning rate schedulers
    scheduler = get_scheduler(optimizer, args, epoch_start)

    # Log file:
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    experiment_name = (
        "params"
        + str(args.batch_size)
        + "_lr="
        + str(args.lr)
        + "_nepochs"
        + str(args.n_epochs)
        + "_nepochsdecay"
        + str(args.n_epochs_decay)
        + "_alpha"
        + str(args.alpha_w)
        + "_beta"
        + str(args.beta_w)
        + "_gamma="
        + str(args.gamma_w)
        + "_gammastartep"
        + str(args.ep_gamma_start)
        + "_data"
        + args.data_to_use
    )
    if args.resume:
        experiment_name = "RESUME" + experiment_name

    log_file_name = os.path.join(args.logdir, experiment_name + ".txt")
    if os.path.isfile(log_file_name):
        log_file = open(log_file_name, "a")
    else:
        log_file = open(log_file_name, "w+")

    log_file.write("\n---------------  " + experiment_name + "  ---------------\n")
    log_file.close()

    exp_log_dir = os.path.join(args.logdir, experiment_name, "")
    if not os.path.exists(exp_log_dir):
        os.makedirs(exp_log_dir)

    global losscount
    global train_mse

    # Run training
    for epoch in range(epoch_start, args.n_epochs + args.n_epochs_decay + 1):
        print(f"\n----- Epoch {epoch} -----")
        if epoch >= args.ep_gamma_start:
            gamma_w = args.gamma_w
            print("epoch ", epoch, "gamma_w is now", gamma_w)

        train_mse = 0.0
        best_val_mse = 99999.0
        losscount = 0

        # Train
        UVDocnet.train()

        for batch in trainloader:
            if args.data_to_use == "both":
                (
                    imgs_doc3D_,
                    imgs_unwarped_doc3D_,
                    grid2D_doc3D_,
                    grid3D_doc3D_,
                ) = batch[0]
                (
                    imgs_UVDoc_,
                    imgs_unwarped_UVDoc_,
                    grid2D_UVDoc_,
                    grid3D_UVDoc_,
                ) = batch[1]
            elif args.data_to_use == "doc3d":
                (
                    imgs_doc3D_,
                    imgs_unwarped_doc3D_,
                    grid2D_doc3D_,
                    grid3D_doc3D_,
                ) = batch

            # Train Doc3D step
            imgs_doc3D = imgs_doc3D_.to(device, non_blocking=True)
            unwarped_GT_doc3D = imgs_unwarped_doc3D_.to(device, non_blocking=True)
            grid2D_GT_doc3D = grid2D_doc3D_.to(device, non_blocking=True)
            grid3D_GT_doc3D = grid3D_doc3D_.to(device, non_blocking=True)

            grid2D_pred_doc3D, grid3D_pred_doc3D = UVDocnet(imgs_doc3D)
            unwarped_pred_doc3D = utils.bilinear_unwarping(imgs_doc3D, grid2D_pred_doc3D, utils.IMG_SIZE)

            optimizer.zero_grad(set_to_none=True)

            recon_loss = criterionL1(unwarped_pred_doc3D, unwarped_GT_doc3D)
            loss_grid2D = criterionL1(grid2D_pred_doc3D, grid2D_GT_doc3D)
            loss_grid3D = criterionL1(grid3D_pred_doc3D, grid3D_GT_doc3D)

            netLoss = args.alpha_w * loss_grid2D + args.beta_w * loss_grid3D + gamma_w * recon_loss
            netLoss.backward()
            optimizer.step()

            tmp_mse = criterionMSE(unwarped_pred_doc3D, unwarped_GT_doc3D)
            train_mse += float(tmp_mse)
            losscount += 1

            # Train UVDoc step
            if args.data_to_use == "both":
                imgs_UVDoc = imgs_UVDoc_.to(device, non_blocking=True)
                unwarped_GT_UVDoc = imgs_unwarped_UVDoc_.to(device, non_blocking=True)
                grid2D_GT_UVDoc = grid2D_UVDoc_.to(device, non_blocking=True)
                grid3D_GT_UVDoc = grid3D_UVDoc_.to(device, non_blocking=True)

                grid2D_pred_UVDoc, grid3D_pred_UVDoc = UVDocnet(imgs_UVDoc)
                unwarped_pred_UVDoc = utils.bilinear_unwarping(imgs_UVDoc, grid2D_pred_UVDoc, utils.IMG_SIZE)

                optimizer.zero_grad(set_to_none=True)

                recon_loss = criterionL1(unwarped_pred_UVDoc, unwarped_GT_UVDoc)
                loss_grid2D = criterionL1(grid2D_pred_UVDoc, grid2D_GT_UVDoc)
                loss_grid3D = criterionL1(grid3D_pred_UVDoc, grid3D_GT_UVDoc)

                netLoss = args.alpha_w * loss_grid2D + args.beta_w * loss_grid3D + gamma_w * recon_loss
                netLoss.backward()
                optimizer.step()

                tmp_mse = criterionMSE(unwarped_pred_UVDoc, unwarped_GT_UVDoc)
                train_mse += float(tmp_mse)
                losscount += 1
            gc.collect()

        train_mse = train_mse / max(1, losscount)
        curr_lr = update_learning_rate(scheduler, optimizer)
        write_log_file(log_file_name, train_mse, epoch + 1, curr_lr, "Train")

        # Evaluation
        UVDocnet.eval()

        with torch.no_grad():
            mse_loss_val = 0.0
            for imgs_val_, imgs_unwarped_val_, _, _ in valloader:
                imgs_val = imgs_val_.to(device)
                unwarped_GT_val = imgs_unwarped_val_.to(device)

                grid2D_pred_val, grid3D_pred_val = UVDocnet(imgs_val)
                unwarped_pred_val = utils.bilinear_unwarping(imgs_val, grid2D_pred_val, utils.IMG_SIZE)

                loss_img_val = criterionMSE(unwarped_pred_val, unwarped_GT_val)
                mse_loss_val += float(loss_img_val)

            val_mse = mse_loss_val / len(valloader)
            write_log_file(log_file_name, val_mse, epoch + 1, curr_lr, "Val")

        # save best models
        if val_mse < best_val_mse or epoch == args.n_epochs + args.n_epochs_decay:
            best_val_mse = val_mse
            state = {
                "epoch": epoch + 1,
                "model_state": UVDocnet.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }
            model_path = exp_log_dir + f"ep_{epoch + 1}_{val_mse:.5f}_{train_mse:.5f}_best_model.pkl"
            torch.save(state, model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparams")

    parser.add_argument(
        "--data_path_doc3D", nargs="?", type=str, default="./data/doc3D/", help="Data path to load Doc3D data."
    )
    parser.add_argument(
        "--data_path_UVDoc", nargs="?", type=str, default="./data/UVDoc/", help="Data path to load UVDoc data."
    )
    parser.add_argument(
        "--data_to_use",
        type=str,
        default="both",
        choices=["both", "doc3d"],
        help="Dataset to use for training, either 'both' for Doc3D and UVDoc, or 'doc3d' for Doc3D only.",
    )
    parser.add_argument("--batch_size", nargs="?", type=int, default=8, help="Batch size.")
    parser.add_argument(
        "--n_epochs",
        nargs="?",
        type=int,
        default=10,
        help="Number of epochs with initial (constant) learning rate.",
    )
    parser.add_argument(
        "--n_epochs_decay",
        nargs="?",
        type=int,
        default=10,
        help="Number of epochs to linearly decay learning rate to zero.",
    )
    parser.add_argument("--lr", nargs="?", type=float, default=0.0002, help="Initial learning rate.")
    parser.add_argument("--alpha_w", nargs="?", type=float, default=5.0, help="Weight for the 2D grid L1 loss.")
    parser.add_argument("--beta_w", nargs="?", type=float, default=5.0, help="Weight for the 3D grid L1 loss.")
    parser.add_argument(
        "--gamma_w", nargs="?", type=float, default=1.0, help="Weight for the image reconstruction loss."
    )
    parser.add_argument(
        "--ep_gamma_start",
        nargs="?",
        type=int,
        default=10,
        help="Epoch from which to start using image reconstruction loss.",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        type=str,
        default=None,
        help="Path to previous saved model to restart from.",
    )
    parser.add_argument("--logdir", nargs="?", type=str, default="./log/default", help="Path to store the logs.")
    parser.add_argument(
        "-a",
        "--appearance_augmentation",
        nargs="*",
        type=str,
        default=["visual", "noise", "color"],
        choices=["shadow", "blur", "visual", "noise", "color"],
        help="Appearance augmentations to use.",
    )
    parser.add_argument(
        "-gUVDoc",
        "--geometric_augmentationsUVDoc",
        nargs="*",
        type=str,
        default=["rotate"],
        choices=["rotate", "flip", "perspective"],
        help="Geometric augmentations to use for the UVDoc dataset.",
    )
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers to use for the dataloaders.")

    args = parser.parse_args()
    main_worker(args)
