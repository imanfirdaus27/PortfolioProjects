"""Training / evaluation loops, checkpointing, logging, reproducibility."""
from __future__ import annotations

import csv                    # writes the per-epoch log
import json                   # writes the settings next to the checkpoint
import random                 # Python's own random numbers
import time                   # measures how long an epoch takes
from pathlib import Path

import numpy as np
import torch

from .metrics import SegMetrics    # the scoring code, kept in one place


def set_seed(seed: int) -> None:
    """Same seed everywhere. Without this you cannot compare two runs."""
    random.seed(seed)                  # Python's shuffling
    np.random.seed(seed)               # numpy's
    torch.manual_seed(seed)            # PyTorch on the processor
    torch.cuda.manual_seed_all(seed)   # PyTorch on the graphics card
    # Force the slower but repeatable algorithms, so the same seed really does
    # give the same numbers twice.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Logger:
    """Writes one CSV row per epoch, plus the config as JSON. No hidden state."""

    def __init__(self, out_dir: Path, cfg=None):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)   # make the run folder
        self.csv_path = self.dir / 'log.csv'
        self.fields = None            # column names, fixed by the first row
        # Save the settings beside the results, so any number can be traced
        # back to the run that produced it.
        if cfg is not None:
            (self.dir / 'config.json').write_text(json.dumps(vars(cfg), indent=2, default=str))

    def log(self, row: dict) -> None:
        # The first row decides the columns and writes the header.
        if self.fields is None:
            self.fields = list(row)
            with open(self.csv_path, 'w', newline='') as f:
                csv.DictWriter(f, self.fields).writeheader()
        # Append this epoch. Opening each time means the log survives a crash.
        with open(self.csv_path, 'a', newline='') as f:
            csv.DictWriter(f, self.fields).writerow({k: row.get(k) for k in self.fields})


def save_ckpt(path: Path, model, optimizer, epoch: int, best: float) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # Save the weights, the optimiser state, and where we were, so training can
    # be picked up again rather than started over.
    torch.save({'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch, 'best': best}, path)


def load_ckpt(path: Path, model, optimizer=None):
    # map_location='cpu' lets a file saved on a graphics card open anywhere.
    ck = torch.load(path, map_location='cpu')
    model.load_state_dict(ck['model'])          # put the weights back
    if optimizer is not None and 'optimizer' in ck:
        optimizer.load_state_dict(ck['optimizer'])
    return ck.get('epoch', 0), ck.get('best', 0.0)


# ------------------------------------------------------------------ baseline
def train_one_epoch(model, loader, loss_fn, optimizer, device, modality, scaler=None):
    model.train()                 # switch on training behaviour
    total, n = 0.0, 0             # running loss, and how many chips it covers
    for batch in loader:
        # non_blocking lets the copy to the graphics card overlap with other work.
        x = batch[modality].to(device, non_blocking=True)
        y = batch['label'].to(device, non_blocking=True)
        # Clear the gradients left from the previous batch.
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            # Mixed precision: use half-size numbers where it is safe to.
            with torch.autocast(device_type=device.split(':')[0], dtype=torch.float16):
                loss = loss_fn(model(x), y)
            # Scale the loss up first, so small gradients do not round to zero.
            scaler.scale(loss).backward()
            scaler.step(optimizer)      # update the weights
            scaler.update()             # adjust the scale for next time
        else:
            loss = loss_fn(model(x), y)
            loss.backward()             # work out the gradients
            optimizer.step()            # update the weights

        # Weight each batch's loss by its size, so the average is honest even
        # when the last batch is smaller.
        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / max(n, 1)


@torch.no_grad()      # nothing is learned here, so skip the gradient bookkeeping
def evaluate(model, loader, device, modality, loss_fn=None):
    model.eval()                  # switch off dropout and freeze batch norm
    metrics = SegMetrics(device=device)
    total, n = 0.0, 0
    for batch in loader:
        x = batch[modality].to(device, non_blocking=True)
        y = batch['label'].to(device, non_blocking=True)
        logits = model(x)
        if loss_fn is not None:
            total += loss_fn(logits, y).item() * x.size(0)
            n += x.size(0)
        # Add this batch to the running confusion matrix rather than scoring it
        # on its own. Averaging per batch would let a chip with three water
        # pixels count as much as a chip that is half flooded.
        metrics.update(logits, y)
    out = metrics.compute()
    out['loss'] = total / max(n, 1) if loss_fn is not None else None
    return out


# --------------------------------------------------------------- distillation
def train_one_epoch_distill(ts_model, loader, distill_loss, optimizer, device, scaler=None):
    ts_model.student.train()      # the student learns
    ts_model.teacher.eval()       # the teacher only gives opinions
    sums, n = {}, 0               # running total of each loss term
    for batch in loader:
        s1 = batch['s1'].to(device, non_blocking=True)      # radar, for the student
        s2 = batch['s2'].to(device, non_blocking=True)      # optical, for the teacher
        y = batch['label'].to(device, non_blocking=True)    # the real labels
        optimizer.zero_grad(set_to_none=True)

        # Mixed precision has to wrap the forward pass, not only the backward.
        # Without this the two networks run in fp32 while cfg.amp says otherwise,
        # which is both slower and, on a 4 GB card, out of memory.
        if scaler is not None:
            # autocast tells PyTorch to use half-size numbers where it safely
            # can, which halves the memory the two networks need.
            with torch.autocast(device_type=device.split(':')[0], dtype=torch.float16):
                # Run both networks on this batch: the student sees radar, the
                # frozen teacher sees optical.
                s_logits, s_feat, t_logits, t_feat = ts_model(s1, s2)
                # Work out all three losses: labels, teacher answers, teacher shapes.
                parts = distill_loss(s_logits, t_logits, y, s_feat, t_feat)
            # Half-size numbers can round tiny gradients away to nothing, so the
            # scaler multiplies the loss up before working the gradients out.
            scaler.scale(parts['total']).backward()
            scaler.step(optimizer)      # update the student's weights
            scaler.update()             # adjust the multiplier for next time
        else:
            # No mixed precision, usually because this is running on the CPU.
            s_logits, s_feat, t_logits, t_feat = ts_model(s1, s2)
            parts = distill_loss(s_logits, t_logits, y, s_feat, t_feat)
            parts['total'].backward()   # work out the gradients
            optimizer.step()            # update the student's weights

        # Keep a running total of every term, weighted by the batch size.
        for k, v in parts.items():
            sums[k] = sums.get(k, 0.0) + float(v) * s1.size(0)
        n += s1.size(0)
    # Turn the totals back into averages for the epoch.
    return {k: v / max(n, 1) for k, v in sums.items()}


def fit(model, train_loader, val_loader, loss_fn, optimizer, device, cfg,
        out_dir: Path, modality: str, scheduler=None, distill_loss=None):
    """Shared training driver for both the baseline and the distillation runs."""
    logger = Logger(out_dir, cfg)
    # Mixed precision only exists on a graphics card, and only if asked for.
    scaler = torch.cuda.amp.GradScaler() if (cfg.amp and device.startswith('cuda')) else None
    best = -1.0                        # best validation score so far
    is_distill = distill_loss is not None   # which of the two paths this is

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()               # start the clock for this epoch
        if is_distill:
            tr = train_one_epoch_distill(model, train_loader, distill_loss,
                                         optimizer, device, scaler)
            train_loss = tr['total']
            # Only the student is scored; the teacher is scaffolding.
            eval_model = model.student
        else:
            train_loss = train_one_epoch(model, train_loader, loss_fn,
                                         optimizer, device, modality, scaler)
            eval_model = model

        # The distilled student is always judged on radar alone.
        val = evaluate(eval_model, val_loader, device, 's1' if is_distill else modality, loss_fn)
        if scheduler is not None:
            scheduler.step()           # lower the learning rate as planned

        # One row of the log: what happened this epoch.
        row = {'epoch': epoch, 'train_loss': round(train_loss, 5),
               'val_loss': round(val['loss'], 5) if val['loss'] is not None else None,
               'val_iou_water': round(val['iou_water'], 5),
               'val_miou': round(val['miou'], 5),
               'val_f1': round(val['f1'], 5),
               'secs': round(time.time() - t0, 1)}
        # The distillation run computes three separate losses and then throws
        # two of them away, which makes it impossible to see which one is
        # dominating. Record them.
        extra = ''          # what gets added to the printed line
        if is_distill:
            # seg is how wrong the student is against the real labels,
            # response is how far it is from the teacher's answers, and
            # pairwise is how far it is from the teacher's sense of shape.
            for part in ('seg', 'response', 'pairwise'):
                if part in tr:
                    row[f'loss_{part}'] = round(tr[part], 5)   # save to log.csv
                    extra += f'  {part} {tr[part]:.3f}'        # show on screen

        logger.log(row)                # append to log.csv
        print(f"epoch {epoch:>3}/{cfg.epochs}  loss {train_loss:.4f}{extra}  "
              f"val IoU(water) {val['iou_water']:.4f}  F1 {val['f1']:.4f}  "
              f"({row['secs']}s)")

        # Always keep the latest weights, so a crash costs one epoch at most.
        save_ckpt(out_dir / 'last.pt', eval_model, optimizer, epoch, best)
        # And keep a separate copy of the best epoch seen so far. Selection is
        # on validation, never on test, which is what keeps the final test
        # comparison meaningful.
        if val['iou_water'] > best:
            best = val['iou_water']
            save_ckpt(out_dir / 'best.pt', eval_model, optimizer, epoch, best)
            print(f'         new best IoU {best:.4f} -> best.pt')

    return best
