"""Predict canopy height from a trained model checkpoint."""
import argparse
from pathlib import Path

import numpy as np
import torch
from osgeo import gdal
from mangrove.data import read_pair
from mangrove.model import DualDPT
from mangrove.weights import load_checkpoint


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="Standard or balanced canopy model weights")
    p.add_argument("--s2", required=True, help="Ten-band TIFF or directory of TIFFs")
    p.add_argument("--dem", required=True, help="Paired DEM TIFF or directory")
    p.add_argument("--output", required=True, help="Output TIFF or directory")
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return p


@torch.no_grad()
def predict(args):
    device = torch.device(args.device)
    amp = args.amp and device.type == "cuda"
    checkpoint = load_checkpoint(args.checkpoint)
    if checkpoint.get("format_version") != 7 or checkpoint.get("stage") not in ("standard", "balanced"):
        raise ValueError("Expected a standard or sample-balanced canopy checkpoint")
    if checkpoint.get("smoke_only"):
        raise ValueError("Use a trained canopy checkpoint for prediction")
    model = DualDPT(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    s2_path, dem_path, output = Path(args.s2), Path(args.dem), Path(args.output)
    if s2_path.is_dir():
        if not dem_path.is_dir():
            raise ValueError("Directory prediction requires a DEM directory")
        files = sorted(p for p in s2_path.iterdir() if p.is_file() and p.suffix.lower() in (".tif", ".tiff"))
        if not files:
            raise ValueError(f"No TIFFs found: {s2_path}")
        pairs = [(p, dem_path / p.name, output / p.name) for p in files]
    else:
        pairs = [(s2_path, dem_path, output)]
    for s2_file, dem_file, target in pairs:
        if not s2_file.is_file() or not dem_file.is_file():
            raise FileNotFoundError(f"Missing paired input: {s2_file}, {dem_file}")
        if target.exists():
            raise FileExistsError(f"Output already exists: {target}")
    for s2_file, dem_file, target in pairs:
        a, b, valid, meta = read_pair(s2_file, dem_file, checkpoint["preprocess"])
        with torch.cuda.amp.autocast(enabled=amp):
            values = model(torch.from_numpy(a[None]).to(device), torch.from_numpy(b[None]).to(device))
        prediction = values.float().cpu().numpy()[0, 0]
        if not np.isfinite(prediction[valid]).all():
            raise FloatingPointError(f"Nonfinite prediction: {s2_file}")
        prediction[~valid] = -9999.
        target.parent.mkdir(parents=True, exist_ok=True)
        ds = gdal.GetDriverByName("GTiff").Create(str(target), 512, 512, 1, gdal.GDT_Float32,
                                                options=["COMPRESS=LZW", "TILED=YES"])
        ds.SetGeoTransform(meta["transform"])
        ds.SetProjection(meta["projection"])
        ds.GetRasterBand(1).SetNoDataValue(-9999.)
        ds.GetRasterBand(1).WriteArray(prediction)
        ds.FlushCache()
        ds = None
        print(f"Saved: {target}", flush=True)
    return [str(target) for _, _, target in pairs]

