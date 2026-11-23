from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from osgeo import gdal, osr

gdal.UseExceptions()


def read_raster(path):
    ds = gdal.Open(str(path))
    if ds is None:
        raise FileNotFoundError(str(path))
    if (ds.RasterYSize, ds.RasterXSize) != (512, 512):
        raise ValueError(f"Expected a 512 x 512 raster: {path}")
    array = ds.ReadAsArray().astype(np.float32)
    if array.ndim == 2:
        array = array[None]
    valid = np.isfinite(array).all(axis=0)
    for i in range(ds.RasterCount):
        band = ds.GetRasterBand(i + 1)
        valid &= band.GetMaskBand().ReadAsArray() != 0
        nodata = band.GetNoDataValue()
        if nodata is not None:
            valid &= array[i] != nodata
    meta = {"transform": ds.GetGeoTransform(), "projection": ds.GetProjection()}
    ds = None
    return array, valid, meta


def check_grid(reference, other, path):
    if not np.allclose(reference["transform"], other["transform"], rtol=0, atol=1e-7):
        raise ValueError(f"Pixel grids differ: {path}")
    a, b = reference["projection"], other["projection"]
    if not a or not b:
        raise ValueError(f"Missing coordinate reference system: {path}")
    sa, sb = osr.SpatialReference(), osr.SpatialReference()
    sa.ImportFromWkt(a)
    sb.ImportFromWkt(b)
    if not sa.IsSame(sb):
        raise ValueError(f"Coordinate reference systems differ: {path}")


from functools import lru_cache
@lru_cache(maxsize=64)
def _coordinate_transformer(wkt):
    from pyproj import Transformer
    return Transformer.from_crs(wkt, "EPSG:4326", always_xy=True)


def geographic_coordinates(meta):
    
    gt = meta["transform"]
    yy, xx = np.mgrid[:512, :512].astype(np.float64)
    xx, yy = xx + 0.5, yy + 0.5
    x = gt[0] + xx * gt[1] + yy * gt[2]
    y = gt[3] + xx * gt[4] + yy * gt[5]

    lon, lat = _coordinate_transformer(meta["projection"]).transform(x, y)
    result = np.stack([(lon + 180.0) / 360.0, (lat + 90.0) / 180.0]).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Invalid geographic coordinates")
    if (result < 0).any() or (result > 1).any():
        raise ValueError("Geographic coordinates outside the valid longitude/latitude range")
    return result


def normalize_inputs(s2, dem, valid, meta, prep):
    if s2.shape[0] != prep["s2_channels"] or dem.shape[0] != 1:
        raise ValueError("Unexpected Sentinel-2 or DEM band count")
    if prep["dem_normalization"] != "input_0_1" or prep["coords_normalization"] != "global_0_1":
        raise ValueError("Incompatible preprocessing configuration")
    if prep["s2_scale"] <= 0:
        raise ValueError("Scale factors must be positive")
    mean = np.asarray(prep["s2_mean"], dtype=np.float32).reshape(-1, 1, 1)
    std = np.asarray(prep["s2_std"], dtype=np.float32).reshape(-1, 1, 1)
    if len(mean) not in (1, s2.shape[0]) or len(std) not in (1, s2.shape[0]) or (std <= 0).any():
        raise ValueError("Invalid Sentinel-2 normalization parameters")
    s2 = (s2 / prep["s2_scale"] - mean) / std
    # DEM is supplied already normalized. Do not fit, rescale or clip it.
    valid_dem = dem[:, valid]
    if not np.isfinite(valid_dem).all() or (valid_dem < 0).any() or (valid_dem > 1).any():
        raise ValueError("Supply a DEM already normalized to [0, 1]; valid values are outside this range")
    if prep["use_coords"]:
        dem = np.concatenate([dem, geographic_coordinates(meta)], axis=0)
    s2[:, ~valid], dem[:, ~valid] = 0, 0
    return s2.astype(np.float32), dem.astype(np.float32)


def read_pair(s2_path, dem_path, prep):
    s2, valid_s2, meta = read_raster(s2_path)
    dem, valid_dem, dem_meta = read_raster(dem_path)
    check_grid(meta, dem_meta, dem_path)
    valid = valid_s2 & valid_dem

    mask_band = prep.get("s2_mask_band")
    if mask_band is not None:
        if not 1 <= mask_band <= s2.shape[0]:
            raise ValueError("Coverage band is outside the raster")
        valid &= s2[mask_band - 1] != 0
    s2, dem = normalize_inputs(s2, dem, valid, meta, prep)
    return s2, dem, valid, meta


class CanopyDataset(Dataset):
    
    def __init__(self, root, prep, augment=False, label_nodata=None,
                 high_threshold=20., dem_root=None, balance=False, seed=42):
        self.prep = prep
        self.balance, self.seed, self.epoch = balance, seed, 0
        self.high_threshold = high_threshold
        self.augment, self.label_nodata = augment, label_nodata
        self.root = Path(root)
        self.files = sorted(p for p in (self.root / "s2").glob("*") if p.suffix.lower() in (".tif", ".tiff"))
        self.label_root = self.root / "label"
        self.dem_root = Path(dem_root) if dem_root else self.root / "dem"
        if not self.files:
            raise ValueError(f"No Sentinel-2 tiles found in {self.root}")
        for p in self.files:
            dem_path, label_path = self.dem_root / p.name, self.label_root / p.name
            if not p.exists() or not dem_path.exists() or not label_path.exists():
                raise FileNotFoundError(f"Missing paired raster for {p.name}")

    def _label(self, path):
        label, valid, meta = read_raster(path)
        if label.shape[0] != 1:
            raise ValueError(f"Expected one label band: {path}")
        valid &= label[0] >= 0
        if self.label_nodata is not None:
            valid &= label[0] != self.label_nodata
        return label, valid, meta

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        rng = random
        if isinstance(index, tuple):
            index, item_seed = index
            rng = random.Random(item_seed)
        p = self.files[index]
        s2, dem, valid, meta = read_pair(p, self.dem_root / p.name, self.prep)
        label_path = self.label_root / p.name
        label, label_valid, label_meta = self._label(label_path)
        check_grid(meta, label_meta, label_path)
        input_valid = valid[None].copy()
        valid = (valid & label_valid)[None]
        if self.balance:
            import hashlib
            token = hashlib.sha256(f"{self.seed}:{self.epoch}:{p.name}".encode()).digest()
            sampler = np.random.default_rng(int.from_bytes(token[:8], 'little'))
            high = np.flatnonzero((valid & (label >= self.high_threshold)).ravel())
            low = np.flatnonzero((valid & (label < self.high_threshold)).ravel())
            n = min(len(high), len(low))
            selected = np.zeros(valid.size, dtype=bool)
            if n:
                selected[sampler.choice(high, n, replace=False)] = True
                selected[sampler.choice(low, n, replace=False)] = True
            valid = selected.reshape(valid.shape)
        label[:, ~valid[0]] = 0
        arrays = [s2, dem, label, valid, input_valid]
        if self.augment:
            k, flip = rng.randrange(4), rng.choice([False, True])
            arrays = [np.rot90(a, k, axes=(-2, -1)) for a in arrays]
            if flip:
                arrays = [np.flip(a, axis=-1) for a in arrays]
        values = [torch.from_numpy(np.ascontiguousarray(a)) for a in arrays]
        return dict(zip(("s2", "dem", "label", "valid", "input_valid"), values))
