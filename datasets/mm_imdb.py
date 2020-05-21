import torch
import os
import numpy as np
from torch.utils.data import Dataset
import random
import re
import unicodedata
import string

# %%

glove = []  # {w: vectors[word2idx[w]] for w in words}
all_letters = string.ascii_letters + " .,;'"
fdim = 0


# %%
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, text, label = sample['image'], sample['text'], sample['label']
        return {'image': torch.from_numpy(image.astype(np.float32)),
                'text': text,
                'label': torch.from_numpy(label.astype(np.float32)),
                'textlen': sample['textlen']}


class Normalize(object):
    """Input image cleaning."""

    def __init__(self, mean_vector, std_devs):
        self.mean_vector, self.std_devs = mean_vector, std_devs

    def __call__(self, sample):
        image = sample['image']
        return {'image': self._normalize(image, self.mean_vector, self.std_devs),
                'text': sample['text'],
                'label': sample['label'], 'textlen': sample['textlen']}

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
            print(tensor.size())
            raise TypeError('tensor is not a torch image. Its size is {}.'.format(tensor.size()))
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
        au = sample['text']
        if rval <= self.p_muting:
            vval = random.random()

            if vval <= 0.5:
                im = sample['image'] * 0
            else:
                au = sample['text'] * 0

        return {'image': im, 'text': au, 'label': sample['label'], 'textlen': sample['textlen']}


# %%
class MM_IMDB(Dataset):

    def __init__(self, root_dir='',  # /home/juanma/Documents/Data/MM_IMDB/mmimdb_np
                 transform=None,
                 stage='train',
                 feat_dim=100,
                 average_text=False):
        """
        Args:
            root_dir (string): Directory where data is.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        if stage == 'train':
            self.len_data = 15552
        elif stage == 'test':
            self.len_data = 7799
        elif stage == 'dev':
            self.len_data = 2608

        self.transform = transform
        self.root_dir = root_dir
        self.stage = stage
        self.average_text = average_text

        global fdim
        fdim = feat_dim

    def __len__(self):
        return self.len_data

    def __getitem__(self, idx):

        imagepath = os.path.join(self.root_dir, self.stage, 'image_{:06}.npy'.format(idx))
        labelpath = os.path.join(self.root_dir, self.stage, 'label_{:06}.npy'.format(idx))
        textpath = os.path.join(self.root_dir, self.stage, 'text_{:06}.npy'.format(idx))

        image = np.load(imagepath)
        label = np.load(labelpath)
        text = np.load(textpath)

        if self.average_text:
            text = text.mean(0)

        textlen = text.shape[0]

        sample = {'image': image, 'text': text, 'label': label, 'textlen': textlen}

        if self.transform:
            sample = self.transform(sample)

        return sample


def collate_imdb(list_samples):
    global fdim
    max_text_len = 0
    for sample in list_samples:
        L = len(sample['text'])
        if max_text_len < L:
            max_text_len = L

    list_images = len(list_samples) * [None]
    list_text = len(list_samples) * [None]
    list_labels = len(list_samples) * [None]
    list_textlen = len(list_samples) * [None]

    for i, sample in enumerate(list_samples):
        text_sample_len = len(sample['text'])

        text_i = sample['text'].astype(np.float32)
        padding = np.asarray([fdim * [-10.0]] * (max_text_len - text_sample_len), np.float32)

        list_images[i] = sample['image']
        list_labels[i] = sample['label']
        if padding.shape[0] > 0:
            list_text[i] = torch.from_numpy(np.concatenate((text_i, padding), 0))
        else:
            list_text[i] = torch.from_numpy(text_i)
        list_textlen[i] = sample['textlen']

    images = torch.transpose(torch.stack(list_images), 1, 3)
    text = torch.stack(list_text)
    labels = torch.stack(list_labels)

    return {'image': images, 'text': text, 'label': labels, 'textlen': list_textlen}
