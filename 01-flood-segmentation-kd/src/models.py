"""Models.

One U-Net, used three ways:
  * S1 student   in_channels=2   (Sentinel-1 VV, VH)
  * S2 teacher   in_channels=13  (Sentinel-2 all bands)
  * baselines    either of the above, trained alone

forward() can also return the bottleneck feature map, which is what the
pair-wise distillation loss needs.
"""
from __future__ import annotations

import torch
import torch.nn as nn                # the building blocks of a network
import torch.nn.functional as F      # the same operations as plain functions


def conv_block(cin, cout):
    # Two convolution layers in a row, the standard U-Net step.
    return nn.Sequential(
        # Conv2d slides a 3 by 3 window over the image and learns what to look
        # for. padding=1 keeps the picture the same size. No bias is needed
        # because the batch norm that follows removes any constant offset.
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        # BatchNorm keeps the numbers in a sensible range so training is stable.
        # ReLU throws away negatives, which is what lets the network learn
        # something other than a straight line.
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int = 2, width: int = 32, depth: int = 4):
        super().__init__()           # set up the PyTorch machinery first
        self.depth = depth
        # Channels double at every level going down.
        chans = [width * (2 ** i) for i in range(depth + 1)]     # e.g. 32 64 128 256 512

        # The contracting path: look at the picture, then shrink it.
        self.encoders = nn.ModuleList()
        c_prev = in_channels         # 2 bands for radar, 13 for optical
        for c in chans[:-1]:
            self.encoders.append(conv_block(c_prev, c))
            c_prev = c               # this level's output feeds the next
        # Max pooling halves the width and height by keeping the biggest value
        # in every 2 by 2 square.
        self.pool = nn.MaxPool2d(2)
        # The deepest layer, where the picture is smallest and the features are
        # most abstract. This is what the pair-wise distillation loss compares.
        self.bottleneck = conv_block(chans[-2], chans[-1])

        # The expanding path: grow the picture back to full size.
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 1, -1, -1):           # walk back up the levels
            # ConvTranspose2d doubles the width and height again.
            self.ups.append(nn.ConvTranspose2d(chans[i + 1], chans[i], 2, stride=2))
            # Twice the channels going in, because the matching encoder output
            # is joined on through the skip connection.
            self.decoders.append(conv_block(chans[i] * 2, chans[i]))

        # A 1 by 1 convolution that turns features into one score per class.
        self.head = nn.Conv2d(chans[0], num_classes, 1)

    def forward(self, x, return_features: bool = False):
        skips = []                   # keep each level's output for later
        for enc in self.encoders:
            x = enc(x)               # look at the picture at this size
            skips.append(x)          # save it for the matching decoder
            x = self.pool(x)         # then halve it

        x = self.bottleneck(x)
        feat = x                                   # bottleneck, used for distillation

        # reversed(skips) because the decoder works back up in the opposite order.
        for up, dec, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = up(x)                              # double the size
            # An odd-sized input can leave the two off by a pixel, so nudge
            # them to match before joining.
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            # cat joins the decoder's features with the saved encoder features.
            # This is the skip connection, and it is what keeps the shoreline sharp.
            x = dec(torch.cat([x, skip], dim=1))

        logits = self.head(x)                      # raw score per class per pixel
        # The distillation losses need the bottleneck too; ordinary training does not.
        return (logits, feat) if return_features else logits


def build_model(modality: str, cfg) -> UNet:
    # Radar has two bands, optical has thirteen. Everything else is identical,
    # which is what makes the three runs comparable.
    in_ch = {'s1': 2, 's2': 13}[modality]
    return UNet(in_ch, num_classes=cfg.num_classes, width=cfg.width, depth=cfg.depth)


class TeacherStudent(nn.Module):
    """Wraps the frozen optical teacher and the trainable SAR student.

    Training reads BOTH modalities. Inference reads Sentinel-1 only, which is
    the entire point of the framework.
    """

    def __init__(self, teacher: nn.Module, student: nn.Module, freeze_teacher: bool = True):
        super().__init__()
        self.teacher = teacher
        self.student = student
        if freeze_teacher:
            # requires_grad_(False) stops the teacher from learning: it only
            # gives opinions, it never changes.
            for p in self.teacher.parameters():
                p.requires_grad_(False)
            # eval() also freezes its batch norm statistics.
            self.teacher.eval()

    def forward(self, s1, s2=None):
        # The student always runs, and always returns its bottleneck as well.
        s_logits, s_feat = self.student(s1, return_features=True)
        # No optical image given, so this is deployment: radar alone.
        if s2 is None:
            return s_logits, s_feat, None, None       # inference: SAR only
        # no_grad because nothing about the teacher needs to be learned, and
        # skipping the bookkeeping saves a lot of memory.
        with torch.no_grad():
            t_logits, t_feat = self.teacher(s2, return_features=True)
        return s_logits, s_feat, t_logits, t_feat

    @torch.no_grad()
    def predict(self, s1):
        # What you would call in the field: a flood map from radar only.
        return self.student(s1)
