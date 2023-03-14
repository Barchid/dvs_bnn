from dataclasses import dataclass
from typing import Tuple
from cv2 import transform
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import transforms
from torchvision.transforms import functional
from tonic import transforms as TF
import numpy as np
import random
import tonic
from project.utils.dvs_noises import EventDrop, EventDrop2
from project.utils.transform_dvs import (
    BackgroundActivityNoise,
    ConcatTimeChannels,
    CutMixEvents,
    CutPasteEvent,
    RandomFlipLR,
    RandomFlipPolarity,
    RandomTimeReversal,
    ToFrame,
    get_frame_representation,
    get_sensor_size
)


class DVSTransformRobu:
    def __init__(
        self,
        sensor_size=None,
        timesteps: int = 10,
        transforms_list=[],
        concat_time_channels=True,
        dataset=None,
        severity=1,
    ):
        trans = []

        representation = get_frame_representation(
            sensor_size, timesteps, dataset=dataset
        )
        
        if "occlusion" in transforms_list:
            trans.append(
                CenteredOcclusion(severity=severity, sensor_size=sensor_size)
            )

        if "background_activity" in transforms_list:
            trans.append(
                BackgroundActivityNoise(severity=severity, sensor_size=sensor_size)
            )

        # TENSOR TRANSFORMATION
        trans.append(representation)

        # if 'crop' in transforms_list:
        if "crop" in transforms_list:
            trans.append(
                transforms.RandomResizedCrop(
                    (128, 128), interpolation=transforms.InterpolationMode.NEAREST
                )
            )
        else:
            trans.append(
                transforms.Resize(
                    (128, 128), interpolation=transforms.InterpolationMode.NEAREST
                )
            )

        # finish by concatenating polarity and timesteps
        if concat_time_channels:
            trans.append(ConcatTimeChannels())

            self.transform = transforms.Compose(
                [
                    representation,
                    transforms.Resize(
                        (128, 128), interpolation=transforms.InterpolationMode.NEAREST
                    ),
                    ConcatTimeChannels(),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    representation,
                    transforms.Resize(
                        (128, 128), interpolation=transforms.InterpolationMode.NEAREST
                    ),
                ]
            )

        self.transform = transforms.Compose(trans)

    def __call__(self, X):
        X = self.transform(X)
        return X


@dataclass
class CenteredOcclusion:
    severity: int
    sensor_size: Tuple[int, int, int]

    def __call__(self, events):
        if self.sensor_size is None:
            sensor_size = get_sensor_size(events)
        else:
            sensor_size = self.sensor_size
            
        c = [.35, .45, 0.50, 0.60, 0.70][self.severity - 1]  # c is the sigma here
        mid = (sensor_size[0] // 2, sensor_size[1] // 2, sensor_size[2])

        occ_len_x = int(sensor_size[0] * c)
        occ_len_y = int(sensor_size[1] * c)

        # get coordinates of a centered crop
        coordinates = []
        for x in range(mid[0] - occ_len_x // 2, mid[0] + occ_len_x // 2):
            for y in range(mid[1] - occ_len_y // 2, mid[1] + occ_len_y // 2):
                coordinates.append((x, y))

        return tonic.transforms.functional.drop_pixel_numpy(events=events, coordinates=coordinates)
    

@dataclass(frozen=True)
class DynamicRotation:
    degrees: Tuple[float] = (-75, 75)

    def __call__(self, frames: torch.Tensor):  # shape (..., H, W)
        timesteps = frames.shape[0]
        angle = float(
            torch.empty(1)
            .uniform_(float(self.degrees[0]), float(self.degrees[1]))
            .item()
        )
        step_angle = angle / (timesteps - 1)

        current_angle = 0.0
        result = torch.zeros_like(frames)
        for t in range(timesteps):
            result[t] = functional.rotate(frames[t], current_angle)
            current_angle += step_angle

        return result


@dataclass(frozen=True)
class DynamicTranslation:
    translate: Tuple[float] = (0.3, 0.3)

    def __call__(self, frames: torch.Tensor):  # shape (T, C, H, W)
        timesteps, H, W = frames.shape[0], frames.shape[-2], frames.shape[-1]

        # compute max translation
        max_dx = float(self.translate[0] * H)
        max_dy = float(self.translate[1] * W)
        max_tx = int(round(torch.empty(1).uniform_(-max_dx, max_dx).item()))
        max_ty = int(round(torch.empty(1).uniform_(-max_dy, max_dy).item()))
        # step translation
        step_tx = max_tx / (timesteps - 1)
        current_tx = 0

        step_ty = max_ty / (timesteps - 1)
        current_ty = 0

        result = torch.zeros_like(frames)
        for t in range(timesteps):
            translations = (round(current_tx), round(current_ty))
            result[t] = functional.affine(
                frames[t], 0.0, translate=translations, scale=1.0, shear=0.0, fill=0
            )
            current_tx += step_tx
            current_ty += step_ty

        return result


@dataclass(frozen=True)
class TransRot:
    dyn_tran = DynamicTranslation(translate=(0.05, 0.05))
    dyn_rot = DynamicRotation(degrees=(-7,7))
    stat_tran = transforms.RandomAffine(0, translate=(0.05, 0.5))
    stat_rot = transforms.RandomRotation(7)

    def __call__(self, frames: torch.Tensor):
        choice = np.random.randint(0, 5)
        if choice == 0:
            return frames
        if choice == 1:
            return self.dyn_tran(frames)
        if choice == 2:
            return self.dyn_rot(frames)
        if choice == 3:
            return self.stat_tran(frames)
        if choice == 4:
            return self.stat_rot(frames)


@dataclass(frozen=True)
class Cutout:
    size: Tuple[float] = (0.3, 0.6)
    nb_holes: int = 3

    def __call__(self, frames: torch.Tensor):  # shape (T, C, H, W)
        timesteps, H, W = frames.shape[0], frames.shape[-2], frames.shape[-1]

        mask = torch.ones_like(frames)
        n_holes = random.randint(1, self.nb_holes)
        for i in range(n_holes):
            # compute size of the
            size = random.uniform(self.size[0], self.size[1])
            size_h = int(H * size)
            size_w = int(W * size)
            x_min, y_min = random.randint(0, W - size_w), random.randint(0, H - size_h)
            x_max, y_max = x_min + size_w, y_min + size_h
            mask[:, :, y_min : (y_max + 1), x_min : (x_max + 1)] = 0.0

        # drop events where the
        frames[mask == 0] = 0.0

        return frames


@dataclass(frozen=True)
class FaceMirror:
    def __call__(self, frames: torch.Tensor):  # shape (T, C, H, W)
        W = frames.shape[-1]
        if random.random() >= 0.5:  # left
            left = frames[:, :, :, 0 : W // 2]
            new_right = functional.hflip(left)
            frames[:, :, :, W // 2 :] = new_right
        else:  # right
            right = frames[:, :, :, W // 2 :]
            new_left = functional.hflip(right)
            frames[:, :, :, 0 : W // 2] = new_left

        return frames
