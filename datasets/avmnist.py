from __future__ import print_function, division
import torch
import os
import numpy as np
from torch.utils.data import Dataset
import random


# %%
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, audio, label = sample['image'], sample['audio'], sample['label']

        return {'image': torch.from_numpy(image.astype(np.float32)),
                'audio': torch.from_numpy(audio.astype(np.float32)),
                'label': int(label)}


class Normalize(object):
    """Input image cleaning."""

    def __init__(self, mean_vector, std_devs):
        self.mean_vector, self.std_devs = mean_vector, std_devs

    def __call__(self, sample):
        image = sample['image']
        audio = sample['audio']

        return {'image': self._normalize(image, self.mean_vector, self.std_devs),
                'audio': audio,
                'label': sample['label']}

    def _normalize(self, tensor, mean, std):
        """Normalize a tensor image with mean and standard deviation.
        See ``Normalize`` for more details.
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
            mean (sequence): Sequence of means for each channel.
            std (sequence): Sequence of standard deviations for each channely.
        Returns:
            Tensor: Normalized Tensor image.
        """
        if not self._is_tensor_image(tensor):
            raise TypeError('tensor is not a torch image.')
        # TODO: make efficient
        for t, m, s in zip(tensor, mean, std):
            t.sub_(m).div_(s)
        return tensor

    def _is_tensor_image(self, img):
        return torch.is_tensor(img) and img.ndimension() == 3


class RandomModalityMuting(object):
    """Randomly turn a mode off."""

    def __init__(self, p_muting=0.1):
        self.p_muting = p_muting

    def __call_(self, sample):
        rval = random.random()

        im = sample['image']
        au = sample['audio']
        if rval <= self.p_muting:
            vval = random.random()

            if vval <= 0.5:
                im = sample['image'] * 0
            else:
                au = sample['audio'] * 0

        return {'image': im, 'audio': au, 'label': sample['label']}


# %%
class AVMnist(Dataset):

    def __init__(self, root_dir='./avMNIST',  # /home/juanma/Documents/Data
                 transform=None,
                 stage='train'):
        """
        Args:
            root_dir (string): Directory where data is.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform = transform
        if stage == 'train':
            self.audio_data = np.load(os.path.join(root_dir, 'audio', 'train_data.npy'))
            self.mnist_data = np.load(os.path.join(root_dir, 'images', 'train_data.npy'))
            self.labels = np.load(os.path.join(root_dir, 'train_labels.npy'))
        else:
            self.audio_data = np.load(os.path.join(root_dir, 'audio', 'test_data.npy'))
            self.mnist_data = np.load(os.path.join(root_dir, 'images', 'test_data.npy'))
            self.labels = np.load(os.path.join(root_dir, 'test_labels.npy'))

        self.audio_data = self.audio_data[:, np.newaxis, :, :]
        self.mnist_data = self.mnist_data.reshape(self.mnist_data.shape[0], 1, 28, 28)

    def __len__(self):
        return self.mnist_data.shape[0]

    def __getitem__(self, idx):

        image = self.mnist_data[idx]
        audio = self.audio_data[idx]
        label = self.labels[idx]

        sample = {'image': image, 'audio': audio, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample
