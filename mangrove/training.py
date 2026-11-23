"""Shared training for standard and sample-balanced canopy height models."""
import argparse
import json
import math
import random
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from mangrove.loader import ThreadedDataLoader
from mangrove.save import AsyncCheckpointer
from mangrove.data import CanopyDataset
from mangrove.model import DualDPT
from mangrove.weights import load_checkpoint, load_s2_weights
from mangrove.losses import JointCanopyLoss


def parser(stage):
    train = argparse.ArgumentParser(description=f"Train the {stage} canopy height model")
    train.set_defaults(stage=stage, auto_run_subdir=True, loss_weights=(1., 1., 1.),
                       loss_max_depth=70., hist_bins=70, sig_warm_iter=100,
                       sig_alpha=10., sig_lambda=0.85, sig_warm_up=False)
    for flag in ("train-root", "val-root", "output-dir"):
        train.add_argument("--" + flag, required=True)
    train.add_argument("--dem-root", help="Optional common DEM directory")
    initial = train.add_mutually_exclusive_group(required=True)
    initial.add_argument("--teacher-checkpoint", help="Ten-channel spectral teacher weights")
    initial.add_argument("--init-checkpoint", help="Standard canopy checkpoint for balanced training")
    initial.add_argument("--resume", help="Resume from last.pth with best.pth beside it")
    train.add_argument("--checkpoint-key", default="teacher")
    train.add_argument("--s2-arch", choices=["small", "base", "large"], default="large")
    train.add_argument("--dem-arch", choices=["small", "base", "large"], default="large")
    train.add_argument("--s2-channels", type=int, choices=[10], default=10)
    train.add_argument("--patch-size", type=int, default=16)
    train.add_argument("--registers", type=int, default=0)
    train.add_argument("--width", type=int, default=128)
    train.add_argument("--use-coords", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument("--s2-scale", type=float, default=1.)
    train.add_argument("--s2-mean", type=float, nargs="+", default=[0.])
    train.add_argument("--s2-std", type=float, nargs="+", default=[1.])
    train.add_argument("--s2-mask-band", type=int, choices=range(1, 11))
    train.add_argument("--label-nodata", type=float, default=0.)
    train.add_argument("--high-threshold", type=float, default=20.)
    train.add_argument("--epochs", type=int, default=10, help="Total epochs, including resumed epochs")
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--workers", type=int, default=0)
    train.add_argument("--io-threads", type=int, default=4, help="Use 0 for PyTorch DataLoader")
    train.add_argument("--encoder-lr", type=float, default=1e-6 if stage == "balanced" else 1e-5)
    train.add_argument("--decoder-lr", type=float, default=1e-5 if stage == "balanced" else 1e-4)
    train.add_argument("--weight-decay", type=float, default=0.01)
    train.add_argument("--grad-clip", type=float, default=1.)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--device", default="cuda")
    train.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument("--multi-gpu", action=argparse.BooleanOptionalAction, default=False)
    train.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=False)
    train.add_argument("--smoke-test", action="store_true")
    train.add_argument("--smoke-save", action="store_true")
    train.add_argument("--benchmark-steps", type=int, default=1)
    return train



def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def worker_init(worker_id):
    seed = torch.initial_seed() % 2 ** 32
    random.seed(seed)
    np.random.seed(seed)


@torch.no_grad()
def evaluate(model, loader, device, amp, high_threshold, max_batches=None):
    """Evaluate supplied validation observations without augmentation or sampling."""
    model.eval()
    squared, absolute, signed, count = 0., 0., 0., 0
    high_squared, high_count = 0., 0
    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        with torch.cuda.amp.autocast(enabled=amp):
            prediction = model(batch['s2'], batch['dem'])
        valid = batch['valid']
        error = (prediction.float() - batch['label'])[valid].double()
        if not torch.isfinite(error).all():
            raise FloatingPointError('Nonfinite validation predictions')
        squared += error.square().sum().item()
        absolute += error.abs().sum().item()
        signed += error.sum().item()
        count += error.numel()
        high = valid & (batch['label'] >= high_threshold)
        high_error = (prediction.float() - batch['label'])[high].double()
        high_squared += high_error.square().sum().item()
        high_count += high_error.numel()
    if not count:
        raise ValueError('No valid observations in the validation TIFFs')
    return dict(val_rmse=math.sqrt(squared/count), val_mae=absolute/count,
                val_me=signed/count, val_pixels=count, val_high_pixels=high_count,
                val_high_rmse=math.sqrt(high_squared/high_count) if high_count else None)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def train_model(args):
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        raise ValueError("Epochs and batch size must be positive; workers must be nonnegative")
    seed_all(args.seed)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.backends.mkldnn.enabled = False
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    amp = args.amp and device.type == "cuda"
    output = Path(args.resume).resolve().parent if args.resume else Path(args.output_dir)
    if getattr(args, "auto_run_subdir", False) and not args.resume:
        output = output / (args.stage + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    args.output_dir = str(output)
    print("Training split:", args.train_root, flush=True)
    print("Validation split:", args.val_root, flush=True)
    print("DEM:", args.dem_root, flush=True)
    print("Encoder weights:", args.teacher_checkpoint or args.resume or args.init_checkpoint, flush=True)
    print("Output directory:", output, flush=True)
    output.mkdir(parents=True, exist_ok=True)
    if not args.resume and any((output / name).exists() for name in ("last.pth", "best.pth", "metrics.jsonl")):
        raise FileExistsError("Output exists; resume or use a new directory")

    saved = load_checkpoint(args.resume or args.init_checkpoint) if args.resume or args.init_checkpoint else None
    if saved:
        if saved.get("smoke_only"):
            raise ValueError("Resume from last.pth, not a smoke-test checkpoint")
        if saved.get("format_version") != 7:
            raise ValueError("Expected a canopy training checkpoint")
        config, prep = saved["model_config"], saved["preprocess"]

        args.label_nodata = saved["label_nodata"]
        if args.resume:
            if args.stage != saved["stage"]:
                raise ValueError("Stage must match the resumed checkpoint")
            args.high_threshold = saved["high_threshold"]
            args.seed = saved["seed"]
        elif args.stage != "balanced" or saved["stage"] != "standard":
            raise ValueError("Initialize the balanced model from a standard canopy checkpoint")
    else:
        if not args.teacher_checkpoint:
            raise ValueError("Supply --teacher-checkpoint, --init-checkpoint or --resume")
        if args.s2_scale is None or args.s2_scale <= 0:
            raise ValueError("A positive Sentinel-2 scale is required")
        config = dict(s2_arch=args.s2_arch, dem_arch=args.dem_arch, s2_channels=args.s2_channels,
                      use_coords=args.use_coords, patch_size=args.patch_size, registers=args.registers, width=args.width, gradient_checkpointing=args.gradient_checkpointing)
        prep = dict(s2_channels=args.s2_channels, use_coords=args.use_coords,
                    s2_scale=args.s2_scale, s2_mean=args.s2_mean, s2_std=args.s2_std,
                    dem_normalization="input_0_1", coords_normalization="global_0_1",
                    s2_mask_band=getattr(args, "s2_mask_band", None))
    if Path(args.train_root).resolve() == Path(args.val_root).resolve():
        raise ValueError("Provide separate training and validation directories")
    if saved:
        loss_config = saved["loss_config"]
    else:
        loss_config = dict(weights=args.loss_weights, max_depth=args.loss_max_depth,
                           bins=args.hist_bins, warm_iter=args.sig_warm_iter,
                           sig_alpha=args.sig_alpha, sig_lambda=args.sig_lambda, sig_warm_up=args.sig_warm_up)
    train_set = CanopyDataset(args.train_root, prep, augment=True,
                              label_nodata=args.label_nodata, dem_root=args.dem_root,
                              balance=args.stage == "balanced",
                              high_threshold=args.high_threshold, seed=args.seed)
    val_set = CanopyDataset(args.val_root, prep, label_nodata=args.label_nodata, dem_root=args.dem_root)
    criterion = JointCanopyLoss(**loss_config).to(device)
    if args.resume:
        criterion.load_state_dict(saved["loss_state"])
    loader_args = dict(batch_size=args.batch_size, num_workers=args.workers,
                       pin_memory=device.type == "cuda", worker_init_fn=worker_init)
    if getattr(args, "io_threads", 0) > 0:
        train_loader = ThreadedDataLoader(train_set, args.batch_size, True, threads=args.io_threads,
                                          pin_memory=device.type=="cuda", seed=args.seed)
        val_loader = ThreadedDataLoader(val_set, args.batch_size, False, threads=args.io_threads,
                                        pin_memory=device.type=="cuda", seed=args.seed)
    else:
        train_loader = DataLoader(train_set, shuffle=True, **loader_args)
        val_loader = DataLoader(val_set, shuffle=False, **loader_args)
    print(f"Training tiles: {len(train_set)}, stage: {args.stage}", flush=True)
    model = DualDPT(**config)
    if saved:
        model.load_state_dict(saved["model"], strict=True)
    else:
        load_s2_weights(model.s2_encoder, args.teacher_checkpoint, args.checkpoint_key)
    model.to(device)
    decoder_params = list(model.s2_decoder.parameters()) + list(model.dem_decoder.parameters()) + list(model.head.parameters())
    optimizer = torch.optim.AdamW([
        {"params": model.dem_encoder.parameters(), "lr": args.encoder_lr},
        {"params": decoder_params, "lr": args.decoder_lr},
    ], weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=amp, init_scale=128.0)
    start, best_score, best_epoch = 0, float("inf"), None
    if args.resume:
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        start = saved["epoch"] + 1
        best_score, best_epoch = saved["best_score"], saved["best_epoch"]
        if not (output / "best.pth").is_file():
            raise FileNotFoundError("The resume directory must contain best.pth")
        restore_rng(saved["rng"])
    del saved
    core_model = model
    gpu_count = 1
    if getattr(args, "multi_gpu", False) and device.type=="cuda" and torch.cuda.device_count()>1:
        gpu_count = torch.cuda.device_count()
        model = torch.nn.DataParallel(core_model, device_ids=list(range(gpu_count)))
    if isinstance(train_loader, ThreadedDataLoader):
        train_loader.epoch = start
    print("GPU count:", gpu_count, "physical global batch:", args.batch_size, flush=True)
    if start >= args.epochs:
        raise ValueError("Checkpoint has already reached the requested epoch count")
    run_info = {"arguments": vars(args), "model_config": config, "preprocess": prep, "loss_config": loss_config}
    (output / "run_config.json").write_text(json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8")

    training_start = time.perf_counter()
    checkpoint_writer = AsyncCheckpointer(output)
    try:
        for epoch in range(start, min(args.epochs, start + 1) if args.smoke_test else args.epochs):
            train_set.epoch = epoch
            model.train()
            epoch_start = time.perf_counter()
            criterion.train()
            component_sums = {}
            loss_sum, selected_count, steps = 0.0, 0, 0
            data_seconds = 0.0
            last_batch_end = time.perf_counter()
            for step, batch in enumerate(train_loader):
                data_seconds += time.perf_counter() - last_batch_end
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                selected = batch["valid"]
                count = int(selected.sum().item())
                if count == 0:
                    continue
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=amp):
                    outputs = model(batch["s2"], batch["dem"], return_aux=True)
                loss, components = criterion(outputs, batch["label"], selected, batch["input_valid"])
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), args.grad_clip)
                if not torch.isfinite(grad_norm):
                    if not amp:
                        raise FloatingPointError("Nonfinite gradients")
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    print("Skipped AMP overflow; new scale:", scaler.get_scale(), flush=True)
                    continue
                scaler.step(optimizer)
                scaler.update()
                for key, value in components.items():
                    component_sums[key] = component_sums.get(key, 0.0) + value.item() * count
                loss_sum += loss.item() * count
                selected_count += count
                steps += 1
                last_batch_end = time.perf_counter()
                if args.smoke_test and steps >= args.benchmark_steps:
                    break
                if step % 20 == 0:
                    status = dict(stage=args.stage, epoch=epoch+1, epochs=args.epochs, batch=step+1,
                                  batches=len(train_loader), batch_size=args.batch_size, loss=loss.item(),
                                  elapsed_seconds=time.perf_counter()-training_start)
                    (output / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
                    print(f"epoch={epoch + 1} batch={step + 1}/{len(train_loader)} loss={loss.item():.4f}", flush=True)
            if steps == 0:
                raise ValueError("No valid training updates")
            train_seconds = time.perf_counter() - epoch_start
            validation_start = time.perf_counter()
            metrics = evaluate(model, val_loader, device, amp, args.high_threshold,
                               max_batches=1 if args.smoke_test else None)
            validation_seconds = time.perf_counter() - validation_start
            score = metrics["val_rmse"]
            improved = score < best_score
            if improved:
                best_score, best_epoch = score, epoch + 1
            record = dict(validation_seconds=validation_seconds, best_val_rmse=best_score, best_epoch=best_epoch, train_seconds=train_seconds, data_wait_seconds=data_seconds, epoch=epoch + 1, stage=args.stage, optimizer_steps=steps,
                          epoch_seconds=time.perf_counter()-epoch_start, batch_size=args.batch_size, gpu_count=gpu_count,
                          peak_gpu_mib=torch.cuda.max_memory_allocated()/1024**2 if device.type=="cuda" else 0, train_loss=loss_sum / selected_count,
                          **{"train_" + k: v / selected_count for k, v in component_sums.items()}, **metrics)
            print(json.dumps(record, ensure_ascii=False), flush=True)
            with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            if args.smoke_test:
                if args.smoke_save:
                    torch.save(dict(format_version=7, model=core_model.state_dict(), model_config=config,
                                    preprocess=prep, stage=args.stage, high_threshold=args.high_threshold,
                                    loss_config=loss_config, label_nodata=args.label_nodata, smoke_only=True),
                               output / "smoke_model.pth")
                print("Smoke test completed", flush=True)
                return str(output / "smoke_model.pth") if args.smoke_save else None
            checkpoint = dict(format_version=7, loss_config=loss_config,
                              loss_state=criterion.state_dict(), model=core_model.state_dict(), model_config=config,
                              preprocess=prep, label_nodata=args.label_nodata, stage=args.stage,
                              high_threshold=args.high_threshold, epoch=epoch, seed=args.seed,
                              best_score=best_score, best_epoch=best_epoch, selection_metric="val_rmse",
                              optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), rng=rng_state())

            checkpoint_writer.save(checkpoint, improved)
    finally:
        checkpoint_writer.close()

    return str(output / "best.pth")


