"""Config. Every number the experiment depends on lives here or in a YAML file.

Nothing in the codebase should contain a magic number. If a reviewer asks
"what learning rate did you use for that table?", the answer is in the
config.json that gets written next to the checkpoint.
"""
from __future__ import annotations

# dataclass writes the boring set-up code for a settings class automatically.
# asdict turns the settings into a plain dictionary, and fields lists the names.
from dataclasses import dataclass, asdict, fields
from pathlib import Path       # handles file paths on any operating system


@dataclass
class Config:
    # data
    root: str = 'data'              # folder holding the downloaded chips
    modality: str = 's1'            # s1 | s2 | both
    batch_size: int = 4             # how many chips go through at once
    num_workers: int = 2            # background helpers that load the files

    # model
    num_classes: int = 2            # water and not water
    width: int = 32                 # channels in the first layer of the U-Net
    depth: int = 4                  # how many times the image is halved

    # optimisation
    epochs: int = 30                # how many passes over the training set
    lr: float = 1e-3                # learning rate, the size of each step
    weight_decay: float = 1e-4      # gentle pull towards smaller weights
    dice_weight: float = 0.5        # how much the overlap loss counts
    class_weights: tuple = (1.0, 5.0)   # water is rare, so weight it up
    amp: bool = True                    # mixed precision, ignored on CPU

    # distillation (only used by `train-distill`)
    alpha: float = 1.0                  # weight on response KD
    beta: float = 1.0                   # weight on pair-wise KD
    temperature: float = 4.0            # how much to soften the teacher's answer
    gate_threshold: float = 0.0         # 0 disables confidence gating
    teacher_ckpt: str = 'runs/teacher_s2/best.pt'   # trained teacher to load

    # bookkeeping
    seed: int = 42                  # fixed so a run can be repeated exactly
    out_dir: str = 'runs/exp'       # where checkpoints and logs are written
    device: str = 'auto'                # auto | cpu | cuda

    def resolve_device(self) -> str:
        # If the user named a device, use it and do not argue.
        if self.device != 'auto':
            return self.device
        import torch
        # Otherwise use the graphics card when there is one, else the processor.
        return 'cuda' if torch.cuda.is_available() else 'cpu'

    def to_dict(self) -> dict:
        # Turn the settings into a dictionary so they can be saved as JSON.
        return asdict(self)

    @classmethod
    def load(cls, path: str | None = None, **overrides) -> 'Config':
        data = {}
        # Start from the YAML file if one was given and it exists.
        if path and Path(path).exists():
            import yaml
            # `or {}` covers an empty file, which yaml reads as None.
            data = yaml.safe_load(Path(path).read_text()) or {}
        # The setting names this class actually knows about.
        known = {f.name for f in fields(cls)}
        # Anything typed on the command line wins over the file, but only if it
        # was really given; None means the user left that option alone.
        data.update({k: v for k, v in overrides.items() if v is not None})
        # A misspelt setting would otherwise be accepted and silently ignored,
        # so refuse anything that is not a real setting name.
        unknown = set(data) - known
        if unknown:
            raise ValueError(f'unknown config keys: {sorted(unknown)}')
        # Build the settings object from the merged values.
        return cls(**data)
