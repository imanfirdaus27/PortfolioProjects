"""Single entry point for the whole project.

    python run.py explore                      look at one chip
    python run.py download --split train       fetch a split
    python run.py stats     --split train      label balance
    python run.py train     --modality s1      baseline
    python run.py train     --modality s2      teacher
    python run.py distill                      proposed framework
    python run.py eval      --ckpt runs/x/best.pt --split test

Every command takes --config configs/default.yaml and any override, e.g.
    python run.py train --modality s1 --epochs 5 --batch-size 2
"""
from __future__ import annotations

import argparse                  # reads the options typed after the script
from pathlib import Path

import torch
from torch.utils.data import DataLoader   # feeds chips to the model in batches

from src.config import Config
from src.data import Sen1Floods11, download_split, class_balance
from src.engine import set_seed, fit, evaluate, load_ckpt
from src.losses import SegLoss, DistillLoss
from src.models import build_model, TeacherStudent


def make_loader(cfg, split, modality, shuffle):
    ds = Sen1Floods11(cfg.root, split=split, modality=modality)
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle,
                      # num_workers loads files in the background so the
                      # graphics card is not left waiting on the disk.
                      num_workers=cfg.num_workers, pin_memory=True,
                      # Drop a final short batch while training, because batch
                      # norm behaves badly on one or two chips. Keep it when
                      # scoring, so every chip counts.
                      drop_last=shuffle)


# ------------------------------------------------------------------ commands
def cmd_explore(cfg, args):
    """Download one chip and plot it. No training, no GPU."""
    # Imported here rather than at the top so the training commands do not have
    # to load matplotlib just to start.
    import urllib.request
    import numpy as np
    import rasterio
    import matplotlib.pyplot as plt
    from src.data import DATA_URL, IGNORE_INDEX, read_split, S1_MIN, S1_MAX, S2_SCALE

    import random
    root = Path(cfg.root)
    split = args.split or 'val'

    def fetch(chip, layers=('S1Hand', 'S2Hand', 'LabelHand')):
        """Download the requested layers of one chip and return them."""
        out = {}
        for m in layers:
            path = root / m / f'{chip}_{m}.tif'
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():                 # only fetch what is missing
                try:
                    urllib.request.urlretrieve(f'{DATA_URL}/{m}/{chip}_{m}.tif', path)
                except Exception as e:
                    raise SystemExit(f'could not fetch {chip} ({m}): {e}\n'
                                     f'Check the chip id exists in one of the splits.')
            with rasterio.open(path) as src:
                out[m] = src.read()
        return out

    if args.chip == 'random':
        # Keep drawing until we land on a chip that actually has labels. Only
        # the label file is fetched while searching - it is small - so we do not
        # waste bandwidth downloading 13-band optical files we then throw away.
        pool = read_split(split, root)
        random.shuffle(pool)
        chip = None
        for candidate in pool[:15]:               # give up after fifteen tries
            lab = fetch(candidate, ['LabelHand'])['LabelHand'][0]
            if (lab != IGNORE_INDEX).any():       # at least one real label
                chip = candidate
                break
        if chip is None:
            chip = pool[0]                        # take the first one anyway
        print(f'chip: {chip}   (random from {split})')
    else:
        chip = args.chip or 'Ghana_1033830'      # a chip that has real labels
        print(f'chip: {chip}')

    layers = fetch(chip)

    s1, s2, lab = layers['S1Hand'], layers['S2Hand'], layers['LabelHand'][0]

    # Report the share of each label value, which is how you spot a cloudy chip.
    names = {-1: 'no data (ignored)', 0: 'not water', 1: 'water'}
    pct = {}
    for v, c in zip(*np.unique(lab, return_counts=True)):
        pct[int(v)] = 100 * c / lab.size
        print(f'  {int(v):>2}  {names[int(v)]:<20} {pct[int(v)]:5.1f}%')

    # A chip that is all no-data is not a bug, so explain it rather than fail.
    if pct.get(IGNORE_INDEX, 0) > 99.9:
        print('\n  NOTE: this chip is entirely no-data. The hand labels were drawn')
        print('  from the optical image, so a fully clouded scene has nothing to')
        print('  label. Nothing is wrong - try another chip:')
        for alt in read_split('val', root)[:6]:
            if alt != chip:
                print(f'      python run.py explore --chip {alt}')
                break

    def st(x):
        # Percentile stretch, so a dark radar image is actually visible.
        x = x.astype(np.float32)
        lo, hi = np.percentile(x, [2, 98])
        return np.clip((x - lo) / (hi - lo + 1e-9), 0, 1)

    # Bands 3, 2 and 1 of Sentinel-2 are red, green and blue.
    rgb = np.dstack([st(s2[3]), st(s2[2]), st(s2[1])])
    mask = np.ma.masked_where(lab < 0, lab)      # hide the unlabelled pixels
    fig, ax = plt.subplots(1, 4, figsize=(18, 5))
    for a, img, title, kw in [
            (ax[0], st(s1[0]), 'Sentinel-1 VV', dict(cmap='gray')),
            (ax[1], st(s1[1]), 'Sentinel-1 VH', dict(cmap='gray')),
            (ax[2], rgb, 'Sentinel-2 RGB', {}),
            (ax[3], mask, f"Label - water {pct.get(1, 0):.1f}%", dict(cmap='bwr', vmin=0, vmax=1))]:
        a.imshow(img, **kw); a.set_title(title); a.set_xticks([]); a.set_yticks([])
    fig.suptitle(chip)
    fig.tight_layout()
    out = Path('figures') / f'{chip}.png'      # pictures go in figures/, models in runs/
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches='tight')
    print(f'saved {out}')
    plt.show()


def cmd_download(cfg, args):
    # Fetch only the layers this modality needs, so an S1 run does not pull
    # down thirteen-band optical files it will never open.
    mods = ['S1Hand', 'LabelHand'] if args.modality == 's1' else \
           ['S2Hand', 'LabelHand'] if args.modality == 's2' else \
           ['S1Hand', 'S2Hand', 'LabelHand']
    # --split accepts a comma-separated list, or does all three by default.
    for split in (args.split.split(',') if args.split else ['train', 'val', 'test']):
        print(f'downloading {split} ...')
        chips = download_split(split, Path(cfg.root), modalities=mods)
        print(f'  {len(chips)} chips ready\n')


def cmd_stats(cfg, args):
    # Count the label pixels. This is where the 9.5 per cent water figure
    # quoted in Chapter 3 comes from.
    ds = Sen1Floods11(cfg.root, split=args.split or 'train', modality='s1')
    print(f'{len(ds)} chips in split {args.split or "train"}')
    for k, (n, pct) in class_balance(ds).items():
        print(f'  label {k:>2}: {n:>12,} px  {pct:5.2f}%')


def cmd_train(cfg, args):
    set_seed(cfg.seed)                    # same seed, so the run can be repeated
    device = cfg.resolve_device()         # graphics card if there is one
    print(f'device: {device} | modality: {cfg.modality}')

    # Shuffle the training chips every epoch; never shuffle validation.
    train_loader = make_loader(cfg, 'train', cfg.modality, True)
    val_loader = make_loader(cfg, 'val', cfg.modality, False)

    model = build_model(cfg.modality, cfg).to(device)      # move it to the GPU
    loss_fn = SegLoss(cfg.dice_weight, list(cfg.class_weights)).to(device)
    # AdamW is Adam with the weight decay handled properly.
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # Cosine annealing lowers the learning rate smoothly to almost nothing by
    # the last epoch, which is why every run settles down near the end.
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.epochs)

    best = fit(model, train_loader, val_loader, loss_fn, optim, device, cfg,
               Path(cfg.out_dir), cfg.modality, sched)
    print(f'best val IoU(water): {best:.4f}')


def cmd_distill(cfg, args):
    set_seed(cfg.seed)
    device = cfg.resolve_device()
    print(f'device: {device} | teacher: {cfg.teacher_ckpt}')

    # Training needs both sensors; validation judges the student on radar alone,
    # which is exactly how the model would be used in the field.
    train_loader = make_loader(cfg, 'train', 'both', True)
    val_loader = make_loader(cfg, 'val', 's1', False)

    teacher = build_model('s2', cfg)
    # Fail early and say what to do, rather than crashing inside torch.load.
    if not Path(cfg.teacher_ckpt).exists():
        raise SystemExit(f'teacher checkpoint not found: {cfg.teacher_ckpt}\n'
                         f'Train it first:  python run.py train --modality s2 '
                         f'--out-dir runs/teacher_s2')
    load_ckpt(cfg.teacher_ckpt, teacher)        # load the trained optical model
    student = build_model('s1', cfg)            # a fresh radar model
    model = TeacherStudent(teacher, student).to(device)

    seg = SegLoss(cfg.dice_weight, list(cfg.class_weights)).to(device)
    # `cfg.gate_threshold or None` turns a threshold of 0 into no gate at all.
    distill = DistillLoss(seg, cfg.alpha, cfg.beta, cfg.temperature,
                          cfg.gate_threshold or None).to(device)
    # Only the student's weights are handed to the optimiser; the teacher is frozen.
    optim = torch.optim.AdamW(student.parameters(), lr=cfg.lr,
                              weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.epochs)

    best = fit(model, train_loader, val_loader, seg, optim, device, cfg,
               Path(cfg.out_dir), 's1', sched, distill_loss=distill)
    print(f'best val IoU(water): {best:.4f}')


def cmd_eval(cfg, args):
    device = cfg.resolve_device()
    model = build_model(cfg.modality, cfg).to(device)
    load_ckpt(args.ckpt, model)                 # load the saved weights
    loader = make_loader(cfg, args.split or 'test', cfg.modality, False)
    m = evaluate(model, loader, device, cfg.modality)
    print(f'\nsplit: {args.split or "test"}   checkpoint: {args.ckpt}')
    for k, v in m.items():
        # Print the decimals to four places, but leave whole numbers alone.
        if v is not None:
            print(f'  {k:<12} {v:.4f}' if isinstance(v, float) else f'  {k:<12} {v}')


# The word typed on the command line, and the function it runs.
COMMANDS = {'explore': cmd_explore, 'download': cmd_download, 'stats': cmd_stats,
            'train': cmd_train, 'distill': cmd_distill, 'eval': cmd_eval}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                # keeps the help text laid out as written above
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # choices=COMMANDS means a typo is rejected with the list of valid words.
    p.add_argument('command', choices=COMMANDS)
    p.add_argument('--config', default='configs/default.yaml')
    p.add_argument('--split')
    p.add_argument('--chip')
    p.add_argument('--ckpt')
    # config overrides
    p.add_argument('--modality', choices=['s1', 's2', 'both'])
    p.add_argument('--epochs', type=int)
    p.add_argument('--batch-size', type=int, dest='batch_size')
    p.add_argument('--lr', type=float)
    p.add_argument('--out-dir', dest='out_dir')
    p.add_argument('--alpha', type=float)
    p.add_argument('--beta', type=float)
    p.add_argument('--gate-threshold', type=float, dest='gate_threshold')
    p.add_argument('--teacher-ckpt', dest='teacher_ckpt')
    p.add_argument('--device')
    args = p.parse_args()

    # Pick out the options that are really settings, and let them override the
    # YAML file. Options left unset are None and are ignored inside Config.load.
    overrides = {k: v for k, v in vars(args).items()
                 if k in {'modality', 'epochs', 'batch_size', 'lr', 'out_dir',
                          'alpha', 'beta', 'gate_threshold', 'teacher_ckpt', 'device'}}
    cfg = Config.load(args.config, **overrides)
    COMMANDS[args.command](cfg, args)      # run the chosen command


if __name__ == '__main__':
    main()
