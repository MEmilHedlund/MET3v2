from typing import Optional, List
import math
import os
import sys

import torch
from torch import Tensor

from util.load_config_files import load_yaml_into_dotdict, dotdict


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def save_checkpoint(folder, filename, model, optimizer, scheduler):
    print(f"[INFO] Saving checkpoint in {folder}/{filename}")
    if not os.path.isdir(folder):
        os.makedirs(folder)
    torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
                }, os.path.join(folder, filename))

def update_logs(logs, key, value):
    if not key in logs:
        logs[key] = [value]
    else:
        logs[key].append(value)
    return logs


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)


def pad_and_nest(batch, ids):
    max_len = max(list(map(len, batch)))
    batch, mask, ids = pad_to_batch_max(batch, ids, max_len)
    nested = NestedTensor(batch, mask.bool())

    return nested, ids

class Prediction:
    def __init__(self, positions=None, velocities=None, bounding_boxes=None, shapes=None, logits=None, uncertainties=None):
        if positions is not None:
            self.positions = positions
        if velocities is not None:
            self.velocities = velocities
        if bounding_boxes is not None:
            self.bounding_boxes = bounding_boxes
        if shapes is not None:
            self.shapes = shapes
        if logits is not None:
            self.logits = logits
        if uncertainties is not None:
            self.uncertainties = uncertainties

        self._states = None

    @property
    def states(self):
        if self.positions is not None and self.velocities is not None:
            return torch.cat((self.positions, self.velocities), dim=2)
        elif self.positions is not None and self.velocities is None:
            return self.positions
        else:
            raise NotImplementedError(f'`states` attribute not implemented for positions {self.positions} and '
                                      f'velocities {self.velocities}.')
    @property
    def bbstates(self):
        if self.bounding_boxes is not None :
            return self.bounding_boxes
        else:
            raise NotImplementedError(f'`states` attribute not implemented for positions {self.positions}')


