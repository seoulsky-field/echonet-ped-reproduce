"""Functions for training and running EF prediction."""

import pickle
import scipy
import math
import os
import time

import click
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
import torch
import torchvision
import tqdm

import echonet


@click.command("joint")
@click.option("--data_dir", type=click.Path(exists=True, file_okay=False), default=None)
@click.option("--output", type=click.Path(file_okay=False), default=None)
@click.option("--ef_model_name", type=click.Choice(
    sorted(name for name in torchvision.models.video.__dict__
           if name.islower() and not name.startswith("__") and callable(torchvision.models.video.__dict__[name]))),
    default="r2plus1d_18")
@click.option("--seg_model_name", type=click.Choice(
    sorted(name for name in torchvision.models.segmentation.__dict__
           if name.islower() and not name.startswith("__") and callable(torchvision.models.segmentation.__dict__[name]))),
    default="deeplabv3_resnet50")
@click.option("--ef_weights", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--seg_weights", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--frames", type=int, default=32)
@click.option("--period", type=int, default=2)
@click.option("--num_workers", type=int, default=4)
@click.option("--batch_size", type=int, default=16)
@click.option("--device", type=str, default=None)
@click.option("--seed", type=int, default=0)
def run(
    data_dir=None,
    output=None,
    ef_model_name="r2plus1d_18",
    seg_model_name="deeplabv3_resnet50",
    ef_weights=None,
    seg_weights=None,
    frames=32,
    period=2,
    num_workers=4,
    batch_size=16,
    device=None,
    seed=0,
):
    """Trains/tests EF prediction model.

    \b
    Args:
        data_dir (str, optional): Directory containing dataset. Defaults to
            `echonet.config.DATA_DIR`.
        output (str, optional): Directory to place outputs. Defaults to
            output/video/<model_name>_<pretrained/random>/.
        task (str, optional): Name of task to predict. Options are the headers
            of FileList.csv. Defaults to ``EF''.
        model_name (str, optional): Name of model. One of ``mc3_18'',
            ``r2plus1d_18'', or ``r3d_18''
            (options are torchvision.models.video.<model_name>)
            Defaults to ``r2plus1d_18''.
        pretrained (bool, optional): Whether to use pretrained weights for model
            Defaults to True.
        weights (str, optional): Path to checkpoint containing weights to
            initialize model. Defaults to None.
        run_test (bool, optional): Whether or not to run on test.
            Defaults to False.
        num_epochs (int, optional): Number of epochs during training.
            Defaults to 45.
        lr (float, optional): Learning rate for SGD
            Defaults to 1e-4.
        weight_decay (float, optional): Weight decay for SGD
            Defaults to 1e-4.
        lr_step_period (int or None, optional): Period of learning rate decay
            (learning rate is decayed by a multiplicative factor of 0.1)
            Defaults to 15.
        frames (int, optional): Number of frames to use in clip
            Defaults to 32.
        period (int, optional): Sampling period for frames
            Defaults to 2.
        n_train_patients (int or None, optional): Number of training patients
            for ablations. Defaults to all patients.
        num_workers (int, optional): Number of subprocesses to use for data
            loading. If 0, the data will be loaded in the main process.
            Defaults to 4.
        device (str or None, optional): Name of device to run on. Options from
            https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device
            Defaults to ``cuda'' if available, and ``cpu'' otherwise.
        batch_size (int, optional): Number of samples to load per batch
            Defaults to 16.
        seed (int, optional): Seed for random number generator. Defaults to 0.
    """

    # Seed RNGs
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Set default output directory
    if output is None:
        output = os.path.join("output", "joint")
    os.makedirs(output, exist_ok=True)

    # Set device for computations
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    # Set up ef model
    ef_model = torchvision.models.video.__dict__[ef_model_name](pretrained=False)
    ef_model.fc = torch.nn.Linear(ef_model.fc.in_features, 1)
    if device.type == "cuda":
        ef_model = torch.nn.DataParallel(ef_model)
    ef_model.to(device)

    if ef_weights is not None:
        checkpoint = torch.load(ef_weights)
        ef_model.load_state_dict(checkpoint['state_dict'])

    # Set up segmentation model
    seg_model = torchvision.models.segmentation.__dict__[seg_model_name](pretrained=False, aux_loss=False)

    seg_model.classifier[-1] = torch.nn.Conv2d(seg_model.classifier[-1].in_channels, 1, kernel_size=seg_model.classifier[-1].kernel_size)  # change number of outputs to 1
    if device.type == "cuda":
        seg_model = torch.nn.DataParallel(seg_model)
    seg_model.to(device)

    if seg_weights is not None:
        checkpoint = torch.load(seg_weights)
        seg_model.load_state_dict(checkpoint['state_dict'])

    # Compute mean and std
    mean, std = echonet.utils.get_mean_and_std(echonet.datasets.Echo(root=data_dir, split="train"))

    # for split in ["val", "test"]:
    for split in ["test"]:
        # Performance with test-time augmentation
        # tasks = ["Filename", "EF", "LargeFrame", "SmallFrame", "LargeTrace", "SmallTrace"]
        try:
            with open(os.path.join(output, "video.pkl"), "rb") as f:
                (y, yhat) = pickle.load(f)
        except FileNotFoundError:
            kwargs = {
                "target_type": "EF",
                "mean": mean,
                "std": std,
                "length": frames,
                "period": period,
            }
            ds = echonet.datasets.Echo(root=data_dir, split=split, **kwargs, clips="all")
            dataloader = torch.utils.data.DataLoader(
                ds, batch_size=1, num_workers=num_workers, shuffle=False, pin_memory=(device.type == "cuda"))
            loss, yhat, y = echonet.utils.video.run_epoch(ef_model, dataloader, False, None, device, save_all=True, block_size=batch_size)
            print("{} (all clips) R2:   {:.3f} ({:.3f} - {:.3f})\n".format(split, *echonet.utils.bootstrap(y, np.array(list(map(lambda x: x.mean(), yhat))), sklearn.metrics.r2_score)))
            print("{} (all clips) MAE:  {:.2f} ({:.2f} - {:.2f})\n".format(split, *echonet.utils.bootstrap(y, np.array(list(map(lambda x: x.mean(), yhat))), sklearn.metrics.mean_absolute_error)))
            print("{} (all clips) RMSE: {:.2f} ({:.2f} - {:.2f})\n".format(split, *tuple(map(math.sqrt, echonet.utils.bootstrap(y, np.array(list(map(lambda x: x.mean(), yhat))), sklearn.metrics.mean_squared_error)))))

            with open(os.path.join(output, "video.pkl"), "wb") as f:
                pickle.dump((y, yhat), f)

        yhat = np.array(list(map(lambda x: x.mean(), yhat)))
        video_ef = yhat

        # Plot actual and predicted EF
        fig = plt.figure(figsize=(3, 3))
        lower = min(y.min(), yhat.min())
        upper = max(y.max(), yhat.max())
        plt.scatter(y, yhat, color="k", s=1, edgecolor=None, zorder=2)
        plt.plot([0, 100], [0, 100], linewidth=1, zorder=3)
        # plt.axis([lower - 3, upper + 3, lower - 3, upper + 3])
        plt.axis([0, 100, 0, 100])
        plt.gca().set_aspect("equal", "box")
        plt.xlabel("Actual EF (%)")
        plt.ylabel("Predicted EF (%)")
        plt.title("Video EF (R2={:.2f})".format(sklearn.metrics.r2_score(y, yhat)))
        plt.xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        plt.grid(color="gainsboro", linestyle="--", linewidth=1, zorder=1)
        plt.tight_layout()
        plt.savefig(os.path.join(output, "video_{}_scatter.pdf".format(split)))
        plt.close(fig)

        # Plot AUROC
        fig = plt.figure(figsize=(3, 3))
        plt.plot([0, 1], [0, 1], linewidth=1, color="k", linestyle="--")
        for thresh in [35, 40, 45, 50]:
            fpr, tpr, _ = sklearn.metrics.roc_curve(y > thresh, yhat)
            print(thresh, sklearn.metrics.roc_auc_score(y > thresh, yhat))
            plt.plot(fpr, tpr)

        plt.axis([-0.01, 1.01, -0.01, 1.01])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.tight_layout()
        plt.savefig(os.path.join(output, "{}_roc.pdf".format(split)))
        plt.close(fig)

        tasks = ["Filename", "EF", "LargeFrame", "SmallFrame", "LargeTrace", "SmallTrace"]
        kwargs = {
            "target_type": tasks,
            "mean": mean,
            "std": std
        }
        dataset = echonet.datasets.Echo(root=data_dir, split=split, **kwargs)
        dataloader = torch.utils.data.DataLoader(dataset,
            batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=(device.type == "cuda"))

        seg_model.eval()
        ef_real = []
        ef_pred = []
        iou_large = []
        iou_small = []
        os.makedirs(os.path.join(output, "disk"), exist_ok=True)
        with torch.no_grad():
            with tqdm.tqdm(total=len(dataloader)) as pbar:
                for (_, (filename, ef, large_frame, small_frame, large_trace, small_trace)) in dataloader:
                    assert not torch.isnan(large_trace).any()
                    assert not torch.isnan(small_trace).any()

                    # Run prediction for diastolic frames and compute loss
                    large_frame = large_frame.to(device)
                    yhat = seg_model(large_frame)["out"]

                    # trace = torch.sigmoid(yhat[:, 0, :, :])
                    # apex = torch.sigmoid(yhat[:, 1, :, :])
                    # base = torch.sigmoid(yhat[:, 2, :, :])
                    trace = yhat[:, 0, :, :]
                    iou_large.extend(np.logical_and(trace.cpu() > 0, large_trace > 0).sum((1, 2)) / np.logical_or(trace.cpu() > 0, large_trace > 0).sum((1, 2)))
                    # apex = yhat[:, 1, :, :]
                    # base = yhat[:, 2, :, :]
                    edv = []
                    for (fn, t) in zip(filename, trace.cpu().numpy()):
                        os.makedirs(os.path.join(output, "disk", os.path.splitext(fn)[0]), exist_ok=True)
                        v, *_ = echonet.utils.volume.calculateVolumeMainAxisTopShift(t, 20, pointShifts=1, output=os.path.join(output, "disk", os.path.splitext(fn)[0], "diastole_computer"))
                        assert len(v.values()) == 1
                        edv.append(list(v.values())[0])
                    # for (fn, t) in zip(filename, large_trace.cpu().numpy()):
                    #     v, *_ = echonet.utils.volume.calculateVolumeMainAxisTopShift(t, 20, pointShifts=1, output=os.path.join(output, "disk", os.path.splitext(fn)[0], "diastole_human"))
                    #     assert len(v.values()) == 1
                        # edv.append(list(v.values())[0])


                    # edv = ((trace > 0).sum(2) ** 2).sum(1)
                    small_frame = small_frame.to(device)
                    yhat = seg_model(small_frame)["out"]

                    # small_trace = small_trace[small_mask]
                    # yhat = yhat.transpose(1, 2)[small_mask]
                    # trace = torch.sigmoid(yhat[:, 0, :, :])
                    # apex = torch.sigmoid(yhat[:, 1, :, :])
                    # base = torch.sigmoid(yhat[:, 2, :, :])
                    trace = yhat[:, 0, :, :]
                    iou_small.extend(np.logical_and(trace.cpu() > 0, large_trace > 0).sum((1, 2)) / np.logical_or(trace.cpu() > 0, large_trace > 0).sum((1, 2)))
                    # apex = yhat[:, 1, :, :]
                    # base = yhat[:, 2, :, :]
                    # trace = trace.cpu().numpy()
                    # trace = small_trace.cpu().numpy()
                    esv = []
                    for (fn, t) in zip(filename, trace.cpu().numpy()):
                        v, *_ = echonet.utils.volume.calculateVolumeMainAxisTopShift(t, 20, pointShifts=1, output=os.path.join(output, "disk", os.path.splitext(fn)[0], "systole_computer"))
                        assert len(v.values()) == 1
                        esv.append(list(v.values())[0])
                    # for (fn, t) in zip(filename, small_trace.cpu().numpy()):
                    #     v, *_ = echonet.utils.volume.calculateVolumeMainAxisTopShift(t, 20, pointShifts=1, output=os.path.join(output, "disk", os.path.splitext(fn)[0], "systole_human"))
                    #     assert len(v.values()) == 1
                        # esv.append(list(v.values())[0])
                    # esv = ((trace > 0).sum(2) ** 2).sum(1)

                    edv = np.array(edv)
                    esv = np.array(esv)
                    # mask = ~np.logical_or(np.isnan(esv), np.isnan(edv))
                    # ef_pred.extend((100 * (1 - esv / edv))[mask])
                    # ef_real.extend(ef.numpy()[mask])
                    ef_pred.extend((100 * (1 - esv / edv)))
                    ef_real.extend(ef.numpy())

                    print("Segmentation EF < 0")
                    for (fn, ef) in zip(filename, 1 - esv / edv):
                        if ef < 0:
                            print(fn)
                    print(80 * "=")

                    pbar.update()

            y = np.array(ef_real)
            yhat = np.array(ef_pred)
            nan_mask = ~np.isnan(yhat)
            y = y[nan_mask]
            yhat = yhat[nan_mask]

            # Plot actual and predicted EF
            fig = plt.figure(figsize=(3, 3))
            lower = min(y.min(), yhat.min())
            upper = max(y.max(), yhat.max())
            plt.scatter(y, yhat, color="k", s=1, edgecolor=None, zorder=2)
            plt.plot([0, 100], [0, 100], linewidth=1, zorder=3)
            # plt.axis([lower - 3, upper + 3, lower - 3, upper + 3])
            plt.axis([0, 100, 0, 100])
            plt.gca().set_aspect("equal", "box")
            plt.xlabel("Actual EF (%)")
            plt.ylabel("Predicted EF (%)")
            # plt.title("Segmentation EF\n(R2={:.2f} on {}/{})".format(sklearn.metrics.r2_score(y, yhat)))
            mask = [0 < e < 100 for e in ef_pred]
            mask = [abs(r - p) < 20 for (r, p) in zip(ef_real, ef_pred)]
            plt.title("Segmentation EF\n(R2={:.2f} on {}/{})".format(
                sklearn.metrics.r2_score([e for (e, m) in zip(ef_real, mask) if m], [e for (e, m) in zip(ef_pred, mask) if m]),
                sum(mask),
                len(ef_pred),
            ))
            plt.xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
            plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
            plt.grid(color="gainsboro", linestyle="--", linewidth=1, zorder=1)
            plt.tight_layout()
            plt.savefig(os.path.join(output, "segmentation_{}_scatter.pdf".format(split)))
            plt.close(fig)

            fig = plt.figure(figsize=(3, 3))
            # lower = min(y.min(), yhat.min())
            # upper = max(y.max(), yhat.max())
            # TODO dice is wrong
            plt.scatter((1 + (np.array(iou_large) + np.array(iou_small))[nan_mask] / 2) / 2, abs(y - yhat), color="k", s=1, edgecolor=None, zorder=2)
            # plt.plot([0, 100], [0, 100], linewidth=1, zorder=3)
            # plt.axis([lower - 3, upper + 3, lower - 3, upper + 3])
            # plt.axis([0, 100, 0, 100])
            # plt.gca().set_aspect("equal", "box")
            plt.xlabel("Dice")
            plt.ylabel("EF Error (%)")
            plt.title("Segmentation vs. EF Error")
            # mask = [0 < e < 100 for e in ef_pred]
            # mask = [abs(r - p) < 20 for (r, p) in zip(ef_real, ef_pred)]
            # plt.title("Segmentation EF\n(R2={:.2f} on {}/{})".format(
            #     sklearn.metrics.r2_score([e for (e, m) in zip(ef_real, mask) if m], [e for (e, m) in zip(ef_pred, mask) if m]),
            #     sum(mask),
            #     len(ef_pred),
            # ))
            # plt.xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
            # plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
            # plt.grid(color="gainsboro", linestyle="--", linewidth=1, zorder=1)
            plt.tight_layout()
            plt.savefig(os.path.join(output, "error_{}_scatter.pdf".format(split)))
            plt.close(fig)

            # Plot video and segmentation EF
            y = video_ef[nan_mask]
            fig = plt.figure(figsize=(3, 3))
            plt.scatter(y, yhat, color="k", s=1, edgecolor=None, zorder=2)
            plt.plot([0, 100], [0, 100], linewidth=1, zorder=3)
            # plt.axis([lower - 3, upper + 3, lower - 3, upper + 3])
            plt.axis([0, 100, 0, 100])
            plt.gca().set_aspect("equal", "box")
            plt.xlabel("Video EF (%)")
            plt.ylabel("Segmentation EF (%)")
            mask = [0 < e < 100 for e in yhat]
            # mask = [abs(r - p) < 20 for (r, p) in zip(ef_real, ef_pred)]
            plt.title("Video vs. Segmentation EF".format(
                # sklearn.metrics.r2_score([e for (e, m) in zip(y, mask) if m], [e for (e, m) in zip(yhat, mask) if m]),
                # sum(mask),
                # len(ef_pred),
            ))
            plt.xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
            plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
            plt.grid(color="gainsboro", linestyle="--", linewidth=1, zorder=1)
            plt.tight_layout()
            plt.savefig(os.path.join(output, "comparison_{}_scatter.pdf".format(split)))
            plt.close(fig)

            mask = [0 < e < 100 for e in ef_pred]
            mask = [abs(r - p) < 20 for (r, p) in zip(ef_real, ef_pred)]
            print(sum(mask), "/", len(mask))
            print(sklearn.metrics.r2_score([e for (e, m) in zip(ef_real, mask) if m], [e for (e, m) in zip(ef_pred, mask) if m]))
            print(scipy.stats.linregress([e for (e, m) in zip(ef_real, mask) if m], [e for (e, m) in zip(ef_pred, mask) if m]))

    return
    # Saving videos with segmentations
    dataset = echonet.datasets.Echo(root=data_dir, split="test",
                                    target_type=["Filename", "LargeIndex", "SmallIndex"],  # Need filename for saving, and human-selected frames to annotate
                                    mean=mean, std=std,  # Normalization
                                    length=None, max_length=None, period=1  # Take all frames
                                    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, num_workers=num_workers, shuffle=False, pin_memory=False, collate_fn=_video_collate_fn)

    # Save videos with segmentation
    if save_video and not all(os.path.isfile(os.path.join(output, "videos", f)) for f in dataloader.dataset.fnames):
        # Only run if missing videos

        model.eval()

        os.makedirs(os.path.join(output, "videos"), exist_ok=True)
        os.makedirs(os.path.join(output, "size"), exist_ok=True)
        echonet.utils.latexify()

        with torch.no_grad():
            with open(os.path.join(output, "size.csv"), "w") as g:
                g.write("Filename,Frame,Size,HumanLarge,HumanSmall,ComputerSmall\n")
                for (x, (filenames, large_index, small_index), length) in tqdm.tqdm(dataloader):
                    # Run segmentation model on blocks of frames one-by-one
                    # The whole concatenated video may be too long to run together
                    y = np.concatenate([model(x[i:(i + batch_size), :, :, :].to(device))["out"].detach().cpu().numpy() for i in range(0, x.shape[0], batch_size)])

                    start = 0
                    x = x.numpy()
                    for (i, (filename, offset)) in enumerate(zip(filenames, length)):
                        # Extract one video and segmentation predictions
                        video = x[start:(start + offset), ...]
                        logit = y[start:(start + offset), 0, :, :]

                        # Un-normalize video
                        video *= std.reshape(1, 3, 1, 1)
                        video += mean.reshape(1, 3, 1, 1)

                        # Get frames, channels, height, and width
                        f, c, h, w = video.shape  # pylint: disable=W0612
                        assert c == 3

                        # Put two copies of the video side by side
                        video = np.concatenate((video, video), 3)

                        # If a pixel is in the segmentation, saturate blue channel
                        # Leave alone otherwise
                        video[:, 0, :, w:] = np.maximum(255. * (logit > 0), video[:, 0, :, w:])  # pylint: disable=E1111

                        # Add blank canvas under pair of videos
                        video = np.concatenate((video, np.zeros_like(video)), 2)

                        # Compute size of segmentation per frame
                        size = (logit > 0).sum((1, 2))

                        # Identify systole frames with peak detection
                        trim_min = sorted(size)[round(len(size) ** 0.05)]
                        trim_max = sorted(size)[round(len(size) ** 0.95)]
                        trim_range = trim_max - trim_min
                        systole = set(scipy.signal.find_peaks(-size, distance=20, prominence=(0.50 * trim_range))[0])

                        # Write sizes and frames to file
                        for (frame, s) in enumerate(size):
                            g.write("{},{},{},{},{},{}\n".format(filename, frame, s, 1 if frame == large_index[i] else 0, 1 if frame == small_index[i] else 0, 1 if frame in systole else 0))

                        # Plot sizes
                        fig = plt.figure(figsize=(size.shape[0] / 50 * 1.5, 3))
                        plt.scatter(np.arange(size.shape[0]) / 50, size, s=1)
                        ylim = plt.ylim()
                        for s in systole:
                            plt.plot(np.array([s, s]) / 50, ylim, linewidth=1)
                        plt.ylim(ylim)
                        plt.title(os.path.splitext(filename)[0])
                        plt.xlabel("Seconds")
                        plt.ylabel("Size (pixels)")
                        plt.tight_layout()
                        plt.savefig(os.path.join(output, "size", os.path.splitext(filename)[0] + ".pdf"))
                        plt.close(fig)

                        # Normalize size to [0, 1]
                        size -= size.min()
                        size = size / size.max()
                        size = 1 - size

                        # Iterate the frames in this video
                        for (f, s) in enumerate(size):

                            # On all frames, mark a pixel for the size of the frame
                            video[:, :, int(round(115 + 100 * s)), int(round(f / len(size) * 200 + 10))] = 255.

                            if f in systole:
                                # If frame is computer-selected systole, mark with a line
                                video[:, :, 115:224, int(round(f / len(size) * 200 + 10))] = 255.

                            def dash(start, stop, on=10, off=10):
                                buf = []
                                x = start
                                while x < stop:
                                    buf.extend(range(x, x + on))
                                    x += on
                                    x += off
                                buf = np.array(buf)
                                buf = buf[buf < stop]
                                return buf
                            d = dash(115, 224)

                            if f == large_index[i]:
                                # If frame is human-selected diastole, mark with green dashed line on all frames
                                video[:, :, d, int(round(f / len(size) * 200 + 10))] = np.array([0, 225, 0]).reshape((1, 3, 1))
                            if f == small_index[i]:
                                # If frame is human-selected systole, mark with red dashed line on all frames
                                video[:, :, d, int(round(f / len(size) * 200 + 10))] = np.array([0, 0, 225]).reshape((1, 3, 1))

                            # Get pixels for a circle centered on the pixel
                            r, c = skimage.draw.disk((int(round(115 + 100 * s)), int(round(f / len(size) * 200 + 10))), 4.1)

                            # On the frame that's being shown, put a circle over the pixel
                            video[f, :, r, c] = 255.

                        # Rearrange dimensions and save
                        video = video.transpose(1, 0, 2, 3)
                        video = video.astype(np.uint8)
                        echonet.utils.savevideo(os.path.join(output, "videos", filename), video, 50)

                        # Move to next video
                        start += offset


def run_epoch(model, dataloader, train, optim, device):
    """Run one epoch of training/evaluation for segmentation.

    Args:
        model (torch.nn.Module): Model to train/evaulate.
        dataloder (torch.utils.data.DataLoader): Dataloader for dataset.
        train (bool): Whether or not to train model.
        optim (torch.optim.Optimizer): Optimizer
        device (torch.device): Device to run on
    """

    total = 0.
    n = 0

    pos = 0
    neg = 0
    pos_pix = 0
    neg_pix = 0

    model.train(train)

    large_inter = 0
    large_union = 0
    small_inter = 0
    small_union = 0
    large_inter_list = []
    large_union_list = []
    small_inter_list = []
    small_union_list = []

    with torch.set_grad_enabled(train):
        with tqdm.tqdm(total=len(dataloader)) as pbar:
            for (_, (large_frame, small_frame, large_trace, small_trace)) in dataloader:

                # Mask out nans (missing trace)
                large_mask = ~torch.isnan(large_trace).any(2).any(1)
                small_mask = ~torch.isnan(small_trace).any(2).any(1)

                large_frame = large_frame[large_mask, :, :]
                large_trace = large_trace[large_mask, :, :]
                small_frame = small_frame[small_mask, :, :]
                small_trace = small_trace[small_mask, :, :]

                assert not torch.isnan(large_frame).any()
                assert not torch.isnan(large_trace).any()
                assert not torch.isnan(small_frame).any()
                assert not torch.isnan(small_trace).any()

                # Count number of pixels in/out of human segmentation
                pos += (large_trace == 1).sum().item()
                pos += (small_trace == 1).sum().item()
                neg += (large_trace == 0).sum().item()
                neg += (small_trace == 0).sum().item()

                # Count number of pixels in/out of computer segmentation
                pos_pix += (large_trace == 1).sum(0).to("cpu").detach().numpy()
                pos_pix += (small_trace == 1).sum(0).to("cpu").detach().numpy()
                neg_pix += (large_trace == 0).sum(0).to("cpu").detach().numpy()
                neg_pix += (small_trace == 0).sum(0).to("cpu").detach().numpy()

                # Run prediction for diastolic frames and compute loss
                large_frame = large_frame.to(device)
                large_trace = large_trace.to(device)
                y_large = model(large_frame)["out"]
                loss_large = torch.nn.functional.binary_cross_entropy_with_logits(y_large[:, 0, :, :], large_trace, reduction="sum")
                # Compute pixel intersection and union between human and computer segmentations
                large_inter += np.logical_and(y_large[:, 0, :, :].detach().cpu().numpy() > 0., large_trace[:, :, :].detach().cpu().numpy() > 0.).sum()
                large_union += np.logical_or(y_large[:, 0, :, :].detach().cpu().numpy() > 0., large_trace[:, :, :].detach().cpu().numpy() > 0.).sum()
                # large_inter_list.extend(np.logical_and(y_large[:, 0, :, :].detach().cpu().numpy() > 0., large_trace[:, :, :].detach().cpu().numpy() > 0.).sum((1, 2)))
                # large_union_list.extend(np.logical_or(y_large[:, 0, :, :].detach().cpu().numpy() > 0., large_trace[:, :, :].detach().cpu().numpy() > 0.).sum((1, 2)))
                index = 0
                for m in large_mask:
                    if m:
                        large_inter_list.append(np.logical_and(y_large[index, 0, :, :].detach().cpu().numpy() > 0., large_trace[index, :, :].detach().cpu().numpy() > 0.).sum())
                        large_union_list.append(np.logical_or(y_large[index, 0, :, :].detach().cpu().numpy() > 0., large_trace[index, :, :].detach().cpu().numpy() > 0.).sum())
                        index += 1
                    else:
                        large_inter_list.append(math.nan)
                        large_union_list.append(math.nan)

                # Run prediction for systolic frames and compute loss
                small_frame = small_frame.to(device)
                small_trace = small_trace.to(device)
                y_small = model(small_frame)["out"]
                loss_small = torch.nn.functional.binary_cross_entropy_with_logits(y_small[:, 0, :, :], small_trace, reduction="sum")
                # Compute pixel intersection and union between human and computer segmentations
                small_inter += np.logical_and(y_small[:, 0, :, :].detach().cpu().numpy() > 0., small_trace[:, :, :].detach().cpu().numpy() > 0.).sum()
                small_union += np.logical_or(y_small[:, 0, :, :].detach().cpu().numpy() > 0., small_trace[:, :, :].detach().cpu().numpy() > 0.).sum()
                # small_inter_list.extend(np.logical_and(y_small[:, 0, :, :].detach().cpu().numpy() > 0., small_trace[:, :, :].detach().cpu().numpy() > 0.).sum((1, 2)))
                # small_union_list.extend(np.logical_or(y_small[:, 0, :, :].detach().cpu().numpy() > 0., small_trace[:, :, :].detach().cpu().numpy() > 0.).sum((1, 2)))
                index = 0
                for m in small_mask:
                    if m:
                        small_inter_list.append(np.logical_and(y_small[index, 0, :, :].detach().cpu().numpy() > 0., small_trace[index, :, :].detach().cpu().numpy() > 0.).sum())
                        small_union_list.append(np.logical_or(y_small[index, 0, :, :].detach().cpu().numpy() > 0., small_trace[index, :, :].detach().cpu().numpy() > 0.).sum())
                        index += 1
                    else:
                        small_inter_list.append(math.nan)
                        small_union_list.append(math.nan)

                # Take gradient step if training
                loss = (loss_large + loss_small) / 2
                if train:
                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                # Accumulate losses and compute baselines
                total += loss.item()
                n += large_trace.size(0)
                p = (pos + 1) / (pos + neg + 2)
                p_pix = (pos_pix + 1) / (pos_pix + neg_pix + 2)

                # Show info on process bar
                pbar.set_postfix_str("{:.4f} ({:.4f}) / {:.4f} {:.4f}, {:.4f}, {:.4f}".format(total / n / 112 / 112, loss.item() / large_trace.size(0) / 112 / 112, -p * math.log(p) - (1 - p) * math.log(1 - p), (-p_pix * np.log(p_pix) - (1 - p_pix) * np.log(1 - p_pix)).mean(), 2 * large_inter / (large_union + large_inter), 2 * small_inter / (small_union + small_inter)))
                pbar.update()

    large_inter_list = np.array(large_inter_list)
    large_union_list = np.array(large_union_list)
    small_inter_list = np.array(small_inter_list)
    small_union_list = np.array(small_union_list)

    return (total / n / 112 / 112,
            large_inter_list,
            large_union_list,
            small_inter_list,
            small_union_list,
            )


def _video_collate_fn(x):
    """Collate function for Pytorch dataloader to merge multiple videos.

    This function should be used in a dataloader for a dataset that returns
    a video as the first element, along with some (non-zero) tuple of
    targets. Then, the input x is a list of tuples:
      - x[i][0] is the i-th video in the batch
      - x[i][1] are the targets for the i-th video

    This function returns a 3-tuple:
      - The first element is the videos concatenated along the frames
        dimension. This is done so that videos of different lengths can be
        processed together (tensors cannot be "jagged", so we cannot have
        a dimension for video, and another for frames).
      - The second element is contains the targets with no modification.
      - The third element is a list of the lengths of the videos in frames.
    """
    video, target = zip(*x)  # Extract the videos and targets

    # ``video'' is a tuple of length ``batch_size''
    #   Each element has shape (channels=3, frames, height, width)
    #   height and width are expected to be the same across videos, but
    #   frames can be different.

    # ``target'' is also a tuple of length ``batch_size''
    # Each element is a tuple of the targets for the item.

    i = list(map(lambda t: t.shape[1], video))  # Extract lengths of videos in frames

    # This contatenates the videos along the the frames dimension (basically
    # playing the videos one after another). The frames dimension is then
    # moved to be first.
    # Resulting shape is (total frames, channels=3, height, width)
    video = torch.as_tensor(np.swapaxes(np.concatenate(video, 1), 0, 1))

    # Swap dimensions (approximately a transpose)
    # Before: target[i][j] is the j-th target of element i
    # After:  target[i][j] is the i-th target of element j
    target = zip(*target)

    return video, target, i
