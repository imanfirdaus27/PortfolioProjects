"""Sen1Floods11 data: download + PyTorch Dataset.

The official hand-labelled split files list pairs like
    Ghana_5079_S1Hand.tif,Ghana_5079_LabelHand.tif
so the chip id is everything before the first '_S1Hand'.
"""
from __future__ import annotations

import csv                    # reads the split files, which are comma separated
import urllib.request         # downloads a file from a web address
from pathlib import Path      # handles file paths

import numpy as np            # does the number work on the images
import rasterio               # opens satellite GeoTIFF files
import torch
from torch.utils.data import Dataset   # the base class PyTorch expects

# Where the dataset lives on Google Cloud Storage.
GCS = 'https://storage.googleapis.com/sen1floods11/v1.1'
DATA_URL = f'{GCS}/data/flood_events/HandLabeled'      # the image chips
SPLIT_URL = f'{GCS}/splits/flood_handlabeled'          # the split lists

# Which file lists the chips for each split.
SPLITS = {
    'train': 'flood_train_data.csv',
    'val': 'flood_valid_data.csv',
    'test': 'flood_test_data.csv',
    'bolivia': 'flood_bolivia_data.csv',   # held-out event, never train on this
}

# Sentinel-1 is in dB. Real values sit roughly in [-50, 1].
S1_MIN, S1_MAX = -50.0, 1.0
# Sentinel-2 L1C is TOA reflectance scaled by 10000.
S2_SCALE = 10000.0

IGNORE_INDEX = -1          # label value that must never contribute to loss or metrics


# --------------------------------------------------------------------- download
def _get(url: str, dst: Path) -> None:
    # Already downloaded and not empty, so there is nothing to do.
    if dst.exists() and dst.stat().st_size > 0:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)   # make the folder first
    urllib.request.urlretrieve(url, dst)            # then fetch the file


def read_split(split: str, root: Path) -> list[str]:
    """Return the list of chip ids for a split, downloading the CSV if needed."""
    # Catch a typo now rather than failing later with a confusing message.
    if split not in SPLITS:
        raise ValueError(f'unknown split {split!r}, expected one of {list(SPLITS)}')
    csv_path = root / 'splits' / SPLITS[split]
    _get(f'{SPLIT_URL}/{SPLITS[split]}', csv_path)   # fetch the list if missing

    chips = []
    with open(csv_path, newline='') as f:
        for row in csv.reader(f):
            # Skip blank lines, then cut off the '_S1Hand.tif' ending so what
            # is left is the chip id shared by all three files of that chip.
            if row and row[0].strip():
                chips.append(row[0].strip().replace('_S1Hand.tif', ''))
    return chips


def download_split(split: str, root: Path, modalities=('S1Hand', 'S2Hand', 'LabelHand'),
                   verbose: bool = True) -> list[str]:
    """Download every chip in a split. Safe to re-run: existing files are skipped."""
    chips = read_split(split, root)
    for i, chip in enumerate(chips, 1):        # start counting at 1, not 0
        for m in modalities:                   # radar, optical, label
            name = f'{chip}_{m}.tif'
            _get(f'{DATA_URL}/{m}/{name}', root / m / name)
        # Report every 25 chips, and again at the end, so it does not look stuck.
        if verbose and (i % 25 == 0 or i == len(chips)):
            print(f'  {split:<8} {i:>4}/{len(chips)} chips')
    return chips


# ---------------------------------------------------------------------- dataset
class Sen1Floods11(Dataset):
    """One item = one 512x512 chip.

    Returns a dict:
        s1    float32 (2, H, W)   scaled to [0, 1]
        s2    float32 (13, H, W)  scaled to [0, 1]
        label int64   (H, W)      values -1 / 0 / 1
        chip  str
    Which tensors are actually loaded depends on `modality`, so an S1-only run
    does not pay the cost of reading 13-band optical files.
    """

    def __init__(self, root, split='train', modality='both', transform=None):
        self.root = Path(root)
        self.split = split
        self.modality = modality
        self.transform = transform        # optional augmentation, unused so far
        self.chips = read_split(split, self.root)
        # Work out once which files this run will actually need.
        self.need_s1 = modality in ('s1', 'both')
        self.need_s2 = modality in ('s2', 'both')

    def __len__(self) -> int:
        # How many chips there are. PyTorch asks for this to plan an epoch.
        return len(self.chips)

    def _read(self, chip: str, layer: str) -> np.ndarray:
        path = self.root / layer / f'{chip}_{layer}.tif'
        # A missing file almost always means the download step was skipped, so
        # say exactly which command fixes it.
        if not path.exists():
            raise FileNotFoundError(
                f'{path} missing. Run:  python run.py download --split {self.split}')
        with rasterio.open(path) as src:
            return src.read()

    def __getitem__(self, idx: int) -> dict:
        # PyTorch calls this once per chip, asking for chip number idx.
        chip = self.chips[idx]
        item = {'chip': chip}

        if self.need_s1:
            s1 = self._read(chip, 'S1Hand').astype(np.float32)
            # Replace anything broken: not-a-number and the two infinities.
            s1 = np.nan_to_num(s1, nan=S1_MIN, posinf=S1_MAX, neginf=S1_MIN)
            # Squeeze the decibel range into 0 to 1. This is Equation 3.1.
            s1 = (np.clip(s1, S1_MIN, S1_MAX) - S1_MIN) / (S1_MAX - S1_MIN)
            item['s1'] = torch.from_numpy(s1)      # hand it to PyTorch

        if self.need_s2:
            # Undo the times-10000 storage to get real reflectance. Equation 3.2.
            s2 = self._read(chip, 'S2Hand').astype(np.float32) / S2_SCALE
            s2 = np.clip(np.nan_to_num(s2), 0.0, 1.0)
            item['s2'] = torch.from_numpy(s2)

        # [0] because the label file has one band. int64 is what the loss wants.
        label = self._read(chip, 'LabelHand')[0].astype(np.int64)
        item['label'] = torch.from_numpy(label)

        # Hook for flips and rotations later; nothing is passed in at present.
        if self.transform is not None:
            item = self.transform(item)
        return item


def class_balance(ds: Sen1Floods11) -> dict:
    """Count label pixels across a dataset. Useful for setting class weights."""
    counts = {-1: 0, 0: 0, 1: 0}          # no data, land, water
    for i in range(len(ds)):
        lab = ds[i]['label'].numpy()
        # unique with return_counts gives each value present and how often.
        v, c = np.unique(lab, return_counts=True)
        for vi, ci in zip(v, c):
            counts[int(vi)] = counts.get(int(vi), 0) + int(ci)
    total = sum(counts.values())
    # Report both the raw count and the share, which is where the 9.5 per cent
    # water figure quoted in Chapter 3 comes from.
    return {k: (v, 100 * v / total) for k, v in counts.items()}
