import os
from typing import Optional
import numpy as np

from tonic.io import read_mnist_file
from tonic.dataset import Dataset
from tonic.download_utils import extract_archive
from tonic import functional as TF
import torch.nn.functional as F
import torch
from torch.utils.data import random_split
import cv2
from celluloid import Camera
import random
import matplotlib.pyplot as plt
import pytorch_lightning as pl


class NCALTECH101Localization(Dataset):
    """N-CALTECH101 dataset <https://www.garrickorchard.com/datasets/n-caltech101>. Events have (xytp) ordering.
    ::

        @article{orchard2015converting,
          title={Converting static image datasets to spiking neuromorphic datasets using saccades},
          author={Orchard, Garrick and Jayawant, Ajinkya and Cohen, Gregory K and Thakor, Nitish},
          journal={Frontiers in neuroscience},
          volume={9},
          pages={437},
          year={2015},
          publisher={Frontiers}
        }

    Parameters:
        save_to (string): Location to save files to on disk.
        transform (callable, optional): A callable of transforms to apply to the data.
        target_transform (callable, optional): A callable of transforms to apply to the targets/labels.
    """

    url = "https://www.dropbox.com/sh/iuv7o3h2gv6g4vd/AADYPdhIBK7g_fPCLKmG6aVpa?dl=1"
    filename = "N-Caltech101-archive.zip"
    file_md5 = "989af2c704103341d616b748b5daa70c"
    data_filename = "Caltech101.zip"
    folder_name = "Caltech101"
    annotation_filename = "Caltech101_annotations.zip"
    annotation_folder = "Caltech101_annotations"

    CLASSES_ENABLED = ["Motorbikes", "Faces_easy", "airplanes", "Leopards",
                       "hawksbill", "chandelier", "ketch", "car_side", "bonsai", "watch"]

    sensor_size = None  # all recordings are of different size
    dtype = np.dtype([("x", int), ("y", int), ("t", int), ("p", int)])
    ordering = dtype.names

    def __init__(self, save_to='data', transform=None):
        super(NCALTECH101Localization, self).__init__(
            save_to, transform=transform, target_transform=None
        )
        if not self._check_exists():
            self.download()
            extract_archive(os.path.join(
                self.location_on_system, self.data_filename))
            extract_archive(os.path.join(
                self.location_on_system, self.annotation_filename))

        self.bboxes = []

        data_path = os.path.join(
            self.location_on_system, NCALTECH101Localization.folder_name)
        anno_path = os.path.join(
            self.location_on_system, NCALTECH101Localization.annotation_folder)
        data_dirs = os.listdir(data_path)
        anno_dirs = os.listdir(anno_path)

        for data_dir in data_dirs:
            # skip if there is no annotation
            if data_dir not in anno_dirs or data_dir not in NCALTECH101Localization.CLASSES_ENABLED:
                continue

            anno_dir = os.path.join(anno_path, data_dir)
            data_dir = os.path.join(data_path, data_dir)

            for data_file in os.listdir(data_dir):
                if not data_file.endswith('bin'):
                    continue

                self.data.append(os.path.join(data_dir, data_file))

                anno_file = os.path.join(
                    anno_dir, data_file.replace('image_', 'annotation_'))
                raw_data = np.fromfile(anno_file, dtype=np.uint16)
                # read x_min, y_min, x_max, y_max from data in the bin file
                self.bboxes.append(
                    np.array([raw_data[2], raw_data[3], raw_data[6], raw_data[7]]))

    def __getitem__(self, index):
        """
        Returns:
            a tuple of (events, target) where target is the index of the target class.
        """
        events = read_mnist_file(self.data[index], dtype=self.dtype)

        bbox = self.bboxes[index]

        # normalize & clip to avoid bugs
        bbox = np.array([
            np.clip(bbox[0], 0, events["x"].max()) / events["x"].max(),
            np.clip(bbox[1], 0, events["y"].max()) / events["y"].max(),
            np.clip(bbox[2], 0, events["x"].max()) / events["x"].max(),
            np.clip(bbox[3], 0, events["y"].max()) / events["y"].max(),
        ]).astype(np.float32)

        if self.transform is not None:
            events = self.transform(events)

        return events, bbox

    def __len__(self):
        return len(self.data)

    def _check_exists(self):
        return self._is_file_present() and self._folder_contains_at_least_n_files_of_type(
            8709, ".bin"
        )


def visualize(dataset):
    frames, bbox, target = random.choice(dataset)
    fig, ax = plt.subplots()  # make it bigger
    camera = Camera(fig)
    for frame in frames:
        on = frame[0, :, :]
        off = frame[1, :, :]
        zer = torch.zeros_like(on)
        frame = torch.stack([on, zer, off]).permute(1, 2, 0).numpy()

        frame = np.ascontiguousarray(frame)

        # image = np.clip(np.ascontiguousarray(
        #         np.copy(image).astype(np.uint8)), a_min=0, a_max=255)
        x_min = int(bbox[0] * 224)
        y_min = int(bbox[1] * 224)
        x_max = int(bbox[2] * 224)
        y_max = int(bbox[3] * 224)
        frame = cv2.rectangle(
            frame, (x_min, y_min), (x_max, y_max), color=(0.5, 0.5, 0.5), thickness=4)

        ax.imshow(frame)
        camera.snap()

    animation = camera.animate(interval=20)
    animation.save('ex.mp4')


def save_caltech_pred(x, y_hat, pl_module: pl.LightningModule, batch_idx: int, output_dir: str, gt):
    fig, ax = plt.subplots()  # make it bigger
    camera = Camera(fig)

    frames = x[:, 1, :, :, :]
    bbox = y_hat[1, :]
    gt_bbox = gt[1, :]

    for frame in frames:
        on = frame[0, :, :]
        off = frame[1, :, :]
        zer = torch.zeros_like(on)
        frame = torch.stack([on, zer, off]).permute(1, 2, 0).numpy()

        frame = np.ascontiguousarray(frame)

        # image = np.clip(np.ascontiguousarray(
        #         np.copy(image).astype(np.uint8)), a_min=0, a_max=255)
        x_min = int(bbox[0] * 224)
        y_min = int(bbox[1] * 224)
        x_max = int(bbox[2] * 224)
        y_max = int(bbox[3] * 224)

        gtx_min = int(gt_bbox[0] * 224)
        gty_min = int(gt_bbox[1] * 224)
        gtx_max = int(gt_bbox[2] * 224)
        gty_max = int(gt_bbox[3] * 224)

        # frame = cv2.rectangle(
        #     frame, (x_min, y_min), (x_max, y_max), color=(0.5, 0.5, 0.5), thickness=2)

        # frame = cv2.rectangle(
        #     frame, (gtx_min, gty_min), (gtx_max, gty_max), color=(0.2, 0.2, 0.2), thickness=2)

        ax.imshow(frame)
        camera.snap()

    animation = camera.animate(interval=100)
    epoch = pl_module.current_epoch
    animation.save(os.path.join(
        output_dir, f"ep{str(epoch).zfill(4)}_batch{str(batch_idx).zfill(6)}.mp4"))

    exit()
