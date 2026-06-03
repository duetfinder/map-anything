# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Training Code for MapAnything.

References:
DUSt3R: https://github.com/naver/dust3r
"""

import datetime
import json
import math
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Sized

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import mapanything.utils.train_tools as train_tools
from mapanything.datasets import get_test_data_loader, get_train_data_loader
from mapanything.models import init_model
from mapanything.train.losses import *  # noqa
from mapanything.utils.inference import (
    build_remote_supervision_views,
    has_joint_remote_supervision,
    loss_of_one_batch_multi_view,
)
from mapanything.utils.train_tools import NativeScalerWithGradNormCount as NativeScaler

# Enable TF32 precision if supported (for GPU >= Ampere and PyTorch >= 1.12)
if hasattr(torch.backends.cuda, "matmul") and hasattr(
    torch.backends.cuda.matmul, "allow_tf32"
):
    torch.backends.cuda.matmul.allow_tf32 = True


def train(args):
    """
    Main training function that handles the entire training process.

    This function initializes the distributed training environment, sets up datasets,
    initializes the model, optimizer, and loss functions, and manages the training
    and evaluation loop across multiple epochs.

    In this training, an epoch is just a chunk of the entire dataset.

    Args:
        args: Configuration object containing all training parameters including
              dataset configs, model configs, training parameters, and loss functions.
    """
    # Initialize distributed training if required
    train_tools.init_distributed_mode(args.distributed)
    global_rank = train_tools.get_rank()
    world_size = train_tools.get_world_size()  # noqa

    # Init output directory and device
    print("output_dir: " + args.output_dir)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Fix the seed
    seed = args.train_params.seed + train_tools.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = not args.train_params.disable_cudnn_benchmark

    # Datasets and Dataloaders
    print("Building train dataset {:s}".format(args.dataset.train_dataset))
    data_loader_train = build_dataset(
        dataset=args.dataset.train_dataset,
        num_workers=args.dataset.num_workers,
        test=False,
        max_num_of_imgs_per_gpu=args.train_params.max_num_of_imgs_per_gpu,
    )
    print("Building test dataset {:s}".format(args.dataset.test_dataset))
    test_batch_size = 2 * (
        args.train_params.max_num_of_imgs_per_gpu // args.dataset.num_views
    )  # Since we don't have any backward overhead
    data_loader_test = {
        dataset.split("(")[0]: build_dataset(
            dataset=dataset,
            num_workers=args.dataset.num_workers,
            test=True,
            batch_size=test_batch_size,
        )
        for dataset in args.dataset.test_dataset.split("+")
        if "(" in dataset
    }

    # Load Model
    if global_rank == 0:
        model = init_model(
            args.model.model_str,
            args.model.model_config,
            torch_hub_force_reload=args.model.torch_hub_force_reload,
        )
    if torch.distributed.is_initialized():
        torch.distributed.barrier()  # Make sure the model is initialized before proceeding
    if global_rank != 0:
        model = init_model(
            args.model.model_str, args.model.model_config, torch_hub_force_reload=False
        )
    model.to(device)  # Move model to device
    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    # Criterion
    print(f">> Creating train criterion = {args.loss.train_criterion}")
    train_criterion = eval(args.loss.train_criterion).to(device)
    print(
        f">> Creating test criterion = {args.loss.test_criterion or args.loss.train_criterion}"
    )
    test_criterion = eval(args.loss.test_criterion or args.loss.train_criterion).to(
        device
    )

    # Load pretrained model if provided
    if args.model.pretrained:
        print("Loading pretrained: ", args.model.pretrained)
        ckpt = torch.load(
            args.model.pretrained, map_location=device, weights_only=False
        )
        print(model.load_state_dict(ckpt["model"], strict=False))
        del ckpt  # in case it occupies memory

    warmstart_ckpt = getattr(args.train_params, "warmstart_ckpt", None)
    if warmstart_ckpt not in (None, "", "None", "none", "null"):
        print("Warm-starting model weights from: ", warmstart_ckpt)
        ckpt = torch.load(warmstart_ckpt, map_location=device, weights_only=False)
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model_state = model.state_dict()
        filtered_state_dict = {}
        skipped_shape_mismatch = []
        for key, value in state_dict.items():
            if key in model_state and hasattr(value, "shape") and value.shape != model_state[key].shape:
                skipped_shape_mismatch.append((key, tuple(value.shape), tuple(model_state[key].shape)))
                continue
            filtered_state_dict[key] = value
        if skipped_shape_mismatch:
            print("Warm-start skipped shape-mismatched keys:")
            for key, old_shape, new_shape in skipped_shape_mismatch[:50]:
                print(f"  {key}: checkpoint{old_shape} -> model{new_shape}")
            if len(skipped_shape_mismatch) > 50:
                print(f"  ... and {len(skipped_shape_mismatch) - 50} more")
        print(model.load_state_dict(filtered_state_dict, strict=False))
        del ckpt, state_dict, filtered_state_dict

    # Init model for DDP training
    if args.distributed.distributed:
        ddp_static_graph = bool(
            getattr(args.train_params, "ddp_static_graph", False)
        )
        ddp_find_unused_parameters = bool(
            getattr(args.train_params, "ddp_find_unused_parameters", True)
        )
        if ddp_static_graph and ddp_find_unused_parameters:
            print(
                "DDP static_graph=True; disabling find_unused_parameters to "
                "avoid duplicate autograd-ready hooks with checkpointed heads."
            )
            ddp_find_unused_parameters = False
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.distributed.gpu],
            find_unused_parameters=ddp_find_unused_parameters,
            static_graph=ddp_static_graph,
        )
        model_without_ddp = model.module

    # Optimizer and loss scaler for gradient accumulation
    # Following timm: set wd as 0 for bias and norm layers
    param_groups, param_groups_name_to_idx_map, param_groups_idx_to_name_map = (
        train_tools.get_parameter_groups(
            model_without_ddp,
            args.train_params.lr,
            args.train_params.weight_decay,
            submodule_configs=args.train_params.submodule_configs,
            warn_not_in_submodule=args.train_params.warn_not_in_submodule,
        )
    )
    optimizer = torch.optim.AdamW(
        param_groups, lr=args.train_params.lr, betas=(0.9, 0.95)
    )
    print(optimizer)
    loss_scaler = NativeScaler()

    def write_log_stats(epoch, train_stats, test_stats):
        """
        Writes training and testing statistics to log files and TensorBoard.

        Args:
            epoch: Current epoch number.
            train_stats: Dictionary containing training metrics.
            test_stats: Dictionary containing testing metrics for each test dataset.
        """
        if train_tools.is_main_process():
            if log_writer is not None:
                log_writer.flush()

            log_stats = dict(
                epoch=epoch, **{f"train_{k}": v for k, v in train_stats.items()}
            )
            for test_name in data_loader_test:
                if test_name not in test_stats:
                    continue
                log_stats.update(
                    {test_name + "_" + k: v for k, v in test_stats[test_name].items()}
                )

            with open(
                os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(log_stats) + "\n")

    def save_model(epoch, fname, best_so_far):
        """
        Saves model checkpoint to disk.

        Args:
            epoch: Current epoch number.
            fname: Filename or identifier for the checkpoint.
            best_so_far: Best validation metric achieved so far.
        """
        train_tools.save_model(
            args=args,
            model_without_ddp=model_without_ddp,
            optimizer=optimizer,
            loss_scaler=loss_scaler,
            epoch=epoch,
            fname=fname,
            best_so_far=best_so_far,
        )

    # Resume from a checkpoint if needed
    last_ckpt_fname = os.path.join(args.output_dir, "checkpoint-last.pth")
    if args.train_params.resume and os.path.isfile(last_ckpt_fname):
        args.train_params.resume_ckpt = last_ckpt_fname
    else:
        args.train_params.resume_ckpt = None
    best_so_far = train_tools.load_model(
        train_args=args.train_params,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
    )
    if best_so_far is None:
        best_so_far = float("inf")

    if global_rank == 0 and args.output_dir is not None:
        log_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        log_writer = None

    print(f"Start training for {args.train_params.epochs} epochs")
    start_time = time.time()
    train_stats = test_stats = {}
    for epoch in range(args.train_params.start_epoch, args.train_params.epochs + 1):
        # Save immediately the last checkpoint
        if epoch > args.train_params.start_epoch:
            if (
                args.train_params.save_freq
                and epoch % args.train_params.save_freq == 0
                or epoch == args.train_params.epochs
            ):
                save_model(epoch - 1, "last", best_so_far)

        # Test on multiple datasets
        new_best = False
        test_stats = {}
        if (
            args.train_params.eval_freq > 0
            and epoch % args.train_params.eval_freq == 0
            and epoch > 0
        ):
            for test_name, testset in data_loader_test.items():
                print(f"Testing on {test_name} ...")
                stats = test_one_epoch(
                    model,
                    test_criterion,
                    testset,
                    device,
                    epoch,
                    log_writer=log_writer,
                    args=args,
                    prefix=test_name,
                )
                test_stats[test_name] = stats

            # Calculate average test loss median
            avg_test_loss_med = np.mean(
                [stats["loss_med"] for stats in test_stats.values()]
            )
            test_stats["Average Test Loss Median"] = avg_test_loss_med
            # Save best
            if avg_test_loss_med < best_so_far:
                best_so_far = avg_test_loss_med
                new_best = True

        # Save more stuff
        write_log_stats(epoch, train_stats, test_stats)

        if epoch > args.train_params.start_epoch:
            if args.train_params.keep_freq and epoch % args.train_params.keep_freq == 0:
                save_model(epoch - 1, str(epoch), best_so_far)
            if new_best:
                save_model(epoch - 1, "best", best_so_far)
        if epoch >= args.train_params.epochs:
            break  # exit after writing last test to disk

        # Train
        train_stats = train_one_epoch(
            model,
            train_criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            log_writer=log_writer,
            args=args,
            param_groups_name_to_idx_map=param_groups_name_to_idx_map,
            param_groups_idx_to_name_map=param_groups_idx_to_name_map,
            model_without_ddp=model_without_ddp,
        )
        if log_writer is not None:
            epoch_1000x = int((epoch + 1) * 1000)
            for name, val in train_stats.items():
                log_writer.add_scalar("train_epoch_" + name, val, epoch_1000x)
            log_writer.flush()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))

    save_final_model(
        args, args.train_params.epochs, model_without_ddp, best_so_far=best_so_far
    )


def save_final_model(args, epoch, model_without_ddp, best_so_far=None):
    """
    Saves the final model checkpoint after training completion.

    Args:
        args: Configuration object containing output directory information.
        epoch: Current epoch number.
        model_without_ddp: Model state dictionary or model instance without DistributedDataParallel wrapper.
        best_so_far: Optional; Best validation metric achieved during training.
    """
    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / "checkpoint-final.pth"
    to_save = {
        "args": args,
        "model": model_without_ddp
        if isinstance(model_without_ddp, dict)
        else model_without_ddp.cpu().state_dict(),
        "epoch": epoch,
    }
    if best_so_far is not None:
        to_save["best_so_far"] = best_so_far
    print(f">> Saving model to {checkpoint_path} ...")
    train_tools.save_on_master(to_save, checkpoint_path)


def build_dataset(
    dataset, num_workers, test, batch_size=None, max_num_of_imgs_per_gpu=None
):
    """
    Builds data loaders for training or testing.

    Args:
        dataset: Dataset specification string.
        num_workers: Number of worker processes for data loading.
        test: Boolean flag indicating whether this is a test dataset.
        batch_size: Number of samples per batch. Defaults to None. Used only for testing.
        max_num_of_imgs_per_gpu: Maximum number of images per GPU. Defaults to None. Used only for training.

    Returns:
        DataLoader: PyTorch DataLoader configured for the specified dataset.
    """
    split = ["Train", "Test"][test]
    print(f"Building {split} Data loader for dataset: ", dataset)
    if test:
        assert batch_size is not None, (
            "batch_size must be specified for testing dataloader"
        )
        loader = get_test_data_loader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_mem=True,
            shuffle=False,
            drop_last=False,
        )
    else:
        assert max_num_of_imgs_per_gpu is not None, (
            "max_num_of_imgs_per_gpu must be specified for training dataloader"
        )
        loader = get_train_data_loader(
            dataset=dataset,
            max_num_of_imgs_per_gpu=max_num_of_imgs_per_gpu,
            num_workers=num_workers,
            pin_mem=True,
            shuffle=True,
            drop_last=True,
        )

    print(f"{split} dataset length: ", len(loader))
    return loader


def _resolve_amp_dtype(use_amp, amp_dtype):
    if not use_amp:
        return torch.float32
    if amp_dtype == "fp16":
        return torch.float16
    if amp_dtype == "bf16":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if amp_dtype == "fp32":
        return torch.float32
    return amp_dtype


def _get_train_param(args, name, default):
    return getattr(args.train_params, name, default)


def _make_remote_control_inputs(remote_gt_views, mode, blank_value):
    control_inputs = []
    for remote_gt_view in remote_gt_views:
        image = remote_gt_view['img']
        if mode == 'same':
            control_image = image
        elif mode == 'blank':
            control_image = torch.full_like(image, fill_value=float(blank_value))
        elif mode == 'shuffled':
            if image.ndim < 1 or image.shape[0] <= 1:
                return None
            control_image = torch.roll(image, shifts=1, dims=0)
        else:
            raise ValueError(f'Unsupported remote control ranking mode: {mode}')
        control_inputs.append({
            'img': control_image,
            'data_norm_type': remote_gt_view['data_norm_type'],
            'instance': 'remote',
        })
    return control_inputs




def _unwrap_train_model(model):
    return model.module if hasattr(model, "module") else model


def _compute_remote_blank_preserve_loss(
    batch,
    model,
    criterion,
    use_amp,
    amp_dtype,
    blank_value,
):
    if not has_joint_remote_supervision(batch):
        return None, {}
    aerial_criterion = getattr(criterion, 'aerial_criterion', None)
    if aerial_criterion is None:
        return None, {}

    remote_gt_views = build_remote_supervision_views(batch)
    if not remote_gt_views:
        return None, {}

    remote_inputs = _make_remote_control_inputs(remote_gt_views, 'blank', blank_value)
    if remote_inputs is None:
        return None, {}

    n_remote = len(remote_inputs)
    amp_dtype = _resolve_amp_dtype(use_amp, amp_dtype)
    with torch.autocast('cuda', enabled=bool(use_amp), dtype=amp_dtype):
        preds = model(list(batch) + remote_inputs)
        aerial_preds = preds[:-n_remote]
        with torch.autocast('cuda', enabled=False):
            preserve_loss, _ = aerial_criterion(list(batch), aerial_preds)

    details = {
        'remote_blank_preserve_aerial_loss': float(preserve_loss.detach()),
    }
    return preserve_loss, details


def _compute_remote_adapter_regularization(model):
    unwrapped = _unwrap_train_model(model)
    if not hasattr(unwrapped, 'get_remote_to_aerial_regularization_terms'):
        return {}
    return unwrapped.get_remote_to_aerial_regularization_terms()

def _compute_remote_control_ranking_loss(
    batch,
    model,
    criterion,
    use_amp,
    amp_dtype,
    margin,
    modes,
    blank_value,
    same_loss,
):
    if same_loss is None or not has_joint_remote_supervision(batch):
        return None, {}
    aerial_criterion = getattr(criterion, 'aerial_criterion', None)
    if aerial_criterion is None:
        return None, {}

    remote_gt_views = build_remote_supervision_views(batch)
    if not remote_gt_views:
        return None, {}

    model_aerial_views = list(batch)
    aerial_gts = list(batch)
    n_remote = len(remote_gt_views)
    amp_dtype = _resolve_amp_dtype(use_amp, amp_dtype)

    def control_aerial_loss_for(control_mode):
        remote_inputs = _make_remote_control_inputs(
            remote_gt_views, control_mode, blank_value
        )
        if remote_inputs is None:
            return None
        model_batch = model_aerial_views + remote_inputs
        was_training = model.training
        try:
            model.eval()
            with torch.no_grad():
                with torch.autocast('cuda', enabled=bool(use_amp), dtype=amp_dtype):
                    preds = model(model_batch)
                    aerial_preds = preds[:-n_remote]
                    with torch.autocast('cuda', enabled=False):
                        control_loss, _ = aerial_criterion(aerial_gts, aerial_preds)
        finally:
            model.train(was_training)
        return control_loss.detach()

    ranking_terms = []
    details = {
        'remote_control_same_aerial_loss': float(same_loss.detach()),
    }
    for mode in modes:
        if mode == 'same':
            continue
        control_loss = control_aerial_loss_for(mode)
        if control_loss is None:
            continue
        term = torch.relu(same_loss - control_loss + float(margin))
        ranking_terms.append(term)
        details[f'remote_control_{mode}_aerial_loss'] = float(control_loss.detach())
        details[f'remote_control_ranking_{mode}_loss'] = float(term.detach())

    if not ranking_terms:
        return None, details

    ranking_loss = torch.stack(ranking_terms).mean()
    details['remote_control_ranking_loss'] = float(ranking_loss.detach())
    return ranking_loss, details


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Sized,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    args,
    log_writer=None,
    param_groups_name_to_idx_map=None,
    param_groups_idx_to_name_map=None,
    model_without_ddp=None,
):
    """
    Trains the model for one epoch.
    Epoch is just a chunk of the entire dataset.

    This function handles the training loop for a single epoch, including forward/backward passes,
    gradient accumulation, learning rate scheduling, and logging metrics.

    Args:
        model: The neural network model to train.
        criterion: Loss function to optimize.
        data_loader: DataLoader providing the training data.
        optimizer: Optimizer for updating model parameters.
        device: Device to run training on (CPU or GPU).
        epoch: Current epoch number.
        loss_scaler: Scaler for gradient accumulation and mixed precision training.
        args: Configuration object containing training parameters.
        log_writer: Optional; TensorBoard SummaryWriter for logging.
        param_groups_name_to_idx_map: Mapping from parameter group names to indices.
        param_groups_idx_to_name_map: Mapping from parameter group indices to names.
        model_without_ddp: Model without DistributedDataParallel wrapper for debugging.

    Returns:
        dict: Dictionary containing training metrics averaged over the epoch.
    """
    model.train(True)
    metric_logger = train_tools.MetricLogger(delimiter="  ")
    for submodule_name in param_groups_name_to_idx_map:
        lr_name = f"lr_{submodule_name}" if submodule_name != "default" else "lr"
        metric_logger.add_meter(
            lr_name, train_tools.SmoothedValue(window_size=1, fmt="{value:.6f}")
        )
    header = "Epoch: [{}]".format(epoch)
    accum_iter = args.train_params.accum_iter

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))

    if hasattr(data_loader, "dataset") and hasattr(data_loader.dataset, "set_epoch"):
        data_loader.dataset.set_epoch(epoch)
    if hasattr(data_loader, "sampler") and hasattr(data_loader.sampler, "set_epoch"):
        data_loader.sampler.set_epoch(epoch)
    if hasattr(data_loader, "batch_sampler") and hasattr(
        data_loader.batch_sampler, "set_epoch"
    ):
        data_loader.batch_sampler.set_epoch(epoch)

    optimizer.zero_grad()

    for data_iter_step, batch in enumerate(
        metric_logger.log_every(data_loader, args.train_params.print_freq, header)
    ):
        n_views = len(batch)
        epoch_f = epoch + data_iter_step / len(data_loader)

        # We use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            train_tools.adjust_learning_rate(
                optimizer,
                epoch_f,
                args.train_params,
                param_groups_idx_to_name_map,
                args.train_params.submodule_configs,
            )

        loss_tuple = loss_of_one_batch_multi_view(
            batch,
            model,
            criterion,
            device,
            use_amp=bool(args.train_params.amp),
            amp_dtype=args.train_params.amp_dtype,
            ret="loss",
            criterion_kwargs={"train_epoch": epoch_f},
        )
        loss, loss_details = loss_tuple  # criterion returns two values
        primary_same_loss = loss

        blank_value = float(_get_train_param(args, "remote_control_blank_value", 0.5))

        adapter_reg_terms = _compute_remote_adapter_regularization(model)
        adapter_reg_weights = {
            "remote_to_aerial_late_gate_l1": float(
                _get_train_param(args, "remote_late_gate_l1_weight", 0.0)
            ),
            "remote_to_aerial_late_gate_l2": float(
                _get_train_param(args, "remote_late_gate_l2_weight", 0.0)
            ),
            "late_weighted_delta_l2": float(
                _get_train_param(args, "remote_late_weighted_delta_l2_weight", 0.0)
            ),
        }
        for reg_name, reg_value in adapter_reg_terms.items():
            loss_details[reg_name] = float(reg_value.detach())
            reg_weight = adapter_reg_weights.get(reg_name, 0.0)
            if reg_weight > 0:
                weighted_reg = reg_weight * reg_value
                loss = loss + weighted_reg
                loss_details[f"{reg_name}_weighted"] = float(weighted_reg.detach())

        blank_preserve_weight = float(
            _get_train_param(args, "remote_blank_preserve_loss_weight", 0.0)
        )
        if blank_preserve_weight > 0:
            preserve_loss, preserve_details = _compute_remote_blank_preserve_loss(
                batch=batch,
                model=model,
                criterion=criterion,
                use_amp=bool(args.train_params.amp),
                amp_dtype=args.train_params.amp_dtype,
                blank_value=blank_value,
            )
            if preserve_loss is not None:
                weighted_preserve_loss = blank_preserve_weight * preserve_loss
                loss = loss + weighted_preserve_loss
                loss_details.update(preserve_details)
                loss_details["remote_blank_preserve_loss_weighted"] = float(
                    weighted_preserve_loss.detach()
                )

        ranking_loss_weight = float(
            _get_train_param(args, "remote_control_ranking_loss_weight", 0.0)
        )
        if ranking_loss_weight > 0:
            ranking_modes = _get_train_param(
                args, "remote_control_ranking_modes", ["blank", "shuffled"]
            )
            if isinstance(ranking_modes, str):
                ranking_modes = [ranking_modes]
            ranking_loss, ranking_details = _compute_remote_control_ranking_loss(
                batch=batch,
                model=model,
                criterion=criterion,
                use_amp=bool(args.train_params.amp),
                amp_dtype=args.train_params.amp_dtype,
                margin=float(_get_train_param(args, "remote_control_ranking_margin", 0.0)),
                modes=ranking_modes,
                blank_value=blank_value,
                same_loss=primary_same_loss,
            )
            if ranking_loss is not None:
                loss = loss + ranking_loss_weight * ranking_loss
                loss_details.update(ranking_details)
                loss_details["remote_control_ranking_loss_weighted"] = float(
                    (ranking_loss_weight * ranking_loss).detach()
                )
        if n_views > 2:
            loss = loss * (
                2 / n_views
            )  # scale the loss relative to the number of views (base is 2 views)
        loss_value = float(loss)

        check_instability = not math.isfinite(loss_value) or (
            args.train_params.check_loss_instability
            and loss_value > args.train_params.max_loss_value
        )
        if check_instability:
            print("Loss is {}, stopping training".format(loss_value), force=True)
            print(f"Loss Details: {loss_details}", force=True)
            print(f"Epoch: {epoch}, Data Iteration: {data_iter_step}", force=True)
            # Save the current batch to the output folder for further inspection
            for view_idx, view in enumerate(batch):
                view_cpu = {}
                for k, v in view.items():
                    view_cpu[k] = v.cpu() if isinstance(v, torch.Tensor) else v
                with open(
                    os.path.join(args.output_dir, f"batch_view_{view_idx}.pkl"), "wb"
                ) as f:
                    pickle.dump(view_cpu, f)
            # Save the model to the output folder for further inspection
            checkpoint_debug_path = os.path.join(
                args.output_dir, "checkpoint-debug.pth"
            )
            to_save_debug = {
                "args": args,
                "model": (
                    model_without_ddp
                    if isinstance(model_without_ddp, dict)
                    else model_without_ddp.cpu().state_dict()
                ),
                "epoch": epoch,
                "data_iter_step": data_iter_step,
            }
            torch.save(to_save_debug, checkpoint_debug_path)
            print(f"Saved debugging material to {args.output_dir}", force=True)
            sys.exit(1)

        # Scale the loss by the number of gradient accumulation iterations
        loss /= accum_iter

        # Compute the scaled gradients (also clip the gradients to max norm of 1)
        gradient_norm = loss_scaler(
            loss,
            optimizer,
            parameters=model.parameters(),
            update_grad=(data_iter_step + 1) % accum_iter == 0,
            clip_grad=1.0,
        )

        # Zero out the gradients to prepare for the next iteration of gradient descent
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        del loss
        del batch

        metric_logger.update(epoch=epoch_f)
        for submodule_name in param_groups_name_to_idx_map:
            lr_name = f"lr_{submodule_name}" if submodule_name != "default" else "lr"
            log_lr = optimizer.param_groups[
                param_groups_name_to_idx_map[submodule_name][0]
            ]["lr"]
            metric_logger.meters[lr_name].update(log_lr)
        metric_logger.update(loss=loss_value, **loss_details)

        if (data_iter_step + 1) % accum_iter == 0 and (
            (data_iter_step + 1) % (accum_iter * args.train_params.print_freq)
        ) == 0:
            loss_value_reduce = train_tools.all_reduce_mean(
                loss_value
            )  # MUST BE EXECUTED BY ALL NODES
            if log_writer is None:
                continue
            """
            We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int(epoch_f * 1000)
            log_writer.add_scalar("train_loss", loss_value_reduce, epoch_1000x)
            if gradient_norm is not None:
                log_writer.add_scalar("train_grad_norm", gradient_norm, epoch_1000x)
            for submodule_name in param_groups_name_to_idx_map:
                lr_name = (
                    f"train_lr_{submodule_name}"
                    if submodule_name != "default"
                    else "train_lr"
                )
                log_lr = optimizer.param_groups[
                    param_groups_name_to_idx_map[submodule_name][0]
                ]["lr"]
                log_writer.add_scalar(lr_name, log_lr, epoch_1000x)
            log_writer.add_scalar("train_iter", epoch_1000x, epoch_1000x)
            for name, val in loss_details.items():
                log_writer.add_scalar("train_" + name, val, epoch_1000x)

    # # Gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    # print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def test_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Sized,
    device: torch.device,
    epoch: int,
    args,
    log_writer=None,
    prefix="test",
):
    """
    Evaluates the model on a test dataset for one epoch.
    Epoch is just a chunk of the entire dataset.

    This function runs evaluation on the test dataset without computing gradients,
    and collects metrics for model performance assessment.

    Args:
        model: The neural network model to evaluate.
        criterion: Loss function for evaluation.
        data_loader: DataLoader providing the test data.
        device: Device to run evaluation on (CPU or GPU).
        epoch: Current epoch number.
        args: Configuration object containing evaluation parameters.
        log_writer: Optional; TensorBoard SummaryWriter for logging.
        prefix: String prefix for logging metrics.

    Returns:
        dict: Dictionary containing evaluation metrics (average and median values).
    """
    model.eval()
    metric_logger = train_tools.MetricLogger(delimiter="  ")
    metric_logger.meters = defaultdict(
        lambda: train_tools.SmoothedValue(window_size=9**9)
    )
    header = "Test Epoch: [{}]".format(epoch)

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))

    if args.train_params.freeze_val_samples_across_all_epochs:
        dataloader_epoch = 0
    else:
        dataloader_epoch = epoch
    if hasattr(data_loader, "dataset") and hasattr(data_loader.dataset, "set_epoch"):
        data_loader.dataset.set_epoch(dataloader_epoch)
    if hasattr(data_loader, "sampler") and hasattr(data_loader.sampler, "set_epoch"):
        data_loader.sampler.set_epoch(dataloader_epoch)
    if hasattr(data_loader, "batch_sampler") and hasattr(
        data_loader.batch_sampler, "set_epoch"
    ):
        data_loader.batch_sampler.set_epoch(dataloader_epoch)

    for _, batch in enumerate(
        metric_logger.log_every(data_loader, args.train_params.print_freq, header)
    ):
        n_views = len(batch)
        loss_tuple = loss_of_one_batch_multi_view(
            batch,
            model,
            criterion,
            device,
            use_amp=bool(args.train_params.amp),
            amp_dtype=args.train_params.amp_dtype,
            ret="loss",
            criterion_kwargs={"train_epoch": float(epoch)},
        )
        loss_value, loss_details = loss_tuple  # criterion returns two values
        if n_views > 2:
            loss_value = loss_value * (
                2 / n_views
            )  # scale the loss relative to the number of views (base is 2 views)
        metric_logger.update(loss=float(loss_value), **loss_details)

    # # Gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    # print("Averaged stats:", metric_logger)

    aggs = [("avg", "global_avg"), ("med", "median")]
    results = {
        f"{k}_{tag}": getattr(meter, attr)
        for k, meter in metric_logger.meters.items()
        for tag, attr in aggs
    }

    if log_writer is not None:
        for name, val in results.items():
            log_writer.add_scalar(prefix + "_" + name, val, 1000 * epoch)

    return results
