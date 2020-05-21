import torch
import torch.nn.functional as F
import os
import numpy as np
from torch.utils.data import Dataset
import random
import cv2
from sklearn.utils import shuffle


# %% tools
def load_video(path, vid_len=24):
    cap = cv2.VideoCapture(path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    heigth = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Init the numpy array
    video = np.zeros((vid_len, width, heigth, 3)).astype(np.float32)
    taken = np.linspace(0, num_frames, vid_len).astype(int)

    np_idx = 0
    for fr_idx in range(num_frames):
        ret, frame = cap.read()

        if cap.isOpened() and fr_idx in taken:
            video[np_idx, :, :, :] = frame.astype(np.float32)
            np_idx += 1

    cap.release()

    return video


# 3d coordinates cf. https://github.com/shahroudy/NTURGB-D/blob/master/Matlab/read_skeleton_file.m for more details
def get_3D_skeleton(path):
    # Read the full content of a file
    with open(path, mode='r') as file:
        content = file.readlines()
    content = [c.strip() for c in content]

    # Nb of frames
    num_frames = int(content[0])

    # Init the numpy array
    np_xyz_coordinates = np.zeros((3, num_frames, 25, 2)).astype(np.float32)
    # Loop over the frames
    i = 1
    for t in range(num_frames):
        # Number of person detected
        nb_person = int(content[i])

        # Loop over the number of person
        for p in range(nb_person):
            i = i + 2
            for j in range(25):
                # Catch the line of j
                i = i + 1
                content_j = content[i]

                # Split the line
                list_content_j = content_j.split(' ')
                list_content_j = [float(c) for c in list_content_j]
                xyz_coordinates = list_content_j[:3]
                # Add in the numpy array
                try:
                    for k in range(3):
                        np_xyz_coordinates[k, t, j, p] = xyz_coordinates[k]
                except Exception as e:
                    pass
                    # print(e)  # 3 persons e.g

        i += 1
    # Replace NaN by 0
    np_xyz_coordinates = np.nan_to_num(np_xyz_coordinates)
    return np_xyz_coordinates


# %%
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        rgb, skel, label = sample['rgb'], sample['ske'], sample['label']
        return {'rgb': torch.from_numpy(rgb.astype(np.float32)),
                'ske': torch.from_numpy(skel.astype(np.float32)),
                'label': torch.from_numpy(np.asarray(label))}


# %%
class NormalizeLen(object):
    """ Return a normalized number of frames. """

    def __init__(self, vid_len=(8, 32)):
        self.vid_len = vid_len

    def __call__(self, sample):
        rgb, skel, label = sample['rgb'], sample['ske'], sample['label']
        if rgb.shape[0] != 1:
            num_frames_rgb = len(rgb)
            indices_rgb = np.linspace(0, num_frames_rgb - 1, self.vid_len[0]).astype(int)
            rgb = rgb[indices_rgb, :, :, :]
        if skel.shape[0] != 1:
            num_frames_skel = skel.shape[1]
            skel = interpole(skel, num_frames_skel, self.vid_len[1])

        return {'rgb': rgb,
                'ske': skel,
                'label': label}


def interpole(data, cropped_length, vid_len):
    C, T, V, M = data.shape
    data = torch.tensor(data, dtype=torch.float)
    data = data.permute(0, 2, 3, 1).contiguous().view(C * V * M, cropped_length)
    data = data[None, :, :, None]
    data = F.interpolate(data, size=(vid_len, 1), mode='bilinear', align_corners=False).squeeze(dim=3).squeeze(dim=0)
    data = data.contiguous().view(C, V, M, vid_len).permute(0, 3, 1, 2).contiguous().numpy()
    return data


# %%

class CenterCrop(object):
    """ Return a temporal crop of given sequences """

    def __init__(self, p_interval=0.9):
        self.p_interval = p_interval

    def __call__(self, sample):
        rgb, skel, label = sample['rgb'], sample['ske'], sample['label']
        if skel.shape[0] != 1:
            valid_size = skel.shape[1]
            bias = int((1 - self.p_interval) * valid_size / 2)
            skel = skel[:, bias:valid_size - bias, :, :]

        if rgb.shape[0] != 1:
            num_frames_rgb = len(rgb)
            bias = int((1 - self.p_interval) * num_frames_rgb / 2)
            rgb = rgb[bias:num_frames_rgb - bias, :, :, :]
        return {'rgb': rgb,
                'ske': skel,
                'label': label}


class AugCrop(object):
    """ Return a temporal crop of given sequences """

    def __init__(self, p_interval=0.5):
        self.p_interval = p_interval

    def __call__(self, sample):
        rgb, skel, label = sample['rgb'], sample['ske'], sample['label']
        ratio = (1.0 - self.p_interval * np.random.rand())
        if rgb.shape[0] != 1:
            num_frames_rgb = len(rgb)
            begin_rgb = (num_frames_rgb - int(num_frames_rgb * ratio)) // 2
            rgb = rgb[begin_rgb:(num_frames_rgb - begin_rgb), :, :, :]

        if skel.shape[0] != 1:
            valid_size = skel.shape[1]
            p = np.random.rand(1) * (1.0 - self.p_interval) + self.p_interval
            cropped_length = np.minimum(np.maximum(int(np.floor(valid_size * p)), 64), valid_size)
            bias = np.random.randint(0, valid_size - cropped_length + 1)
            skel = skel[:, bias:bias + cropped_length, :, :]

        return {'rgb': rgb,
                'ske': skel,
                'label': label}


# %%
class NTU(Dataset):

    def __init__(self, root_dir='',  # /home/juanma/Documents/Data/ROSE_Action
                 transform=None,
                 stage='train',
                 vid_len=(8, 32),
                 vid_dim=256,
                 vid_fr=30,
                 args=None):
        """
        Args:
            root_dir (string): Directory where data is.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        if stage == 'train':
            subjects = [1, 4, 8, 13, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
        elif stage == 'trainexp':
            subjects = [1, 4, 8, 13, 15, 17, 19]
        elif stage == 'test':
            subjects = [3, 6, 7, 10, 11, 12, 20, 21, 22, 23, 24, 26, 29, 30, 32, 33, 36, 37, 39, 40]
        elif stage == 'dev':  # smaller train datase for exploration
            subjects = [2, 5, 9, 14]

        basename_rgb = os.path.join(root_dir, 'nturgbd_rgb/avi_{0}x{0}_{1}'.format(vid_dim, vid_fr))
        basename_ske = os.path.join(root_dir, 'nturgbd_skeletons')

        self.original_w, self.original_h = 1920, 1080
        self.vid_len = vid_len

        self.rgb_list = []
        self.dep_list = []
        self.ske_list = []
        self.labels = []

        self.rgb_list += [os.path.join(basename_rgb, f) for f in sorted(os.listdir(basename_rgb)) if
                          f.split(".")[-1] == "avi" and int(f[9:12]) in subjects]
        self.ske_list += [os.path.join(basename_ske, f) for f in sorted(os.listdir(basename_ske)) if
                          f.split(".")[-1] == "skeleton" and int(f[9:12]) in subjects]
        self.labels += [int(f[17:20]) for f in sorted(os.listdir(basename_rgb)) if
                        f.split(".")[-1] == "avi" and int(f[9:12]) in subjects]

        if args.no_bad_skel:
            with open("bad_skel.txt", "r") as f:
                for line in f.readlines():
                    if os.path.join(basename_ske, line[:-1] + ".skeleton") in self.ske_list:
                        i = self.ske_list.index(os.path.join(basename_ske, line[:-1] + ".skeleton"))
                        self.ske_list.pop(i)
                        self.rgb_list.pop(i)
                        self.labels.pop(i)

        self.rgb_list, self.ske_list, self.labels = shuffle(self.rgb_list, self.ske_list, self.labels)

        self.transform = transform
        self.root_dir = root_dir
        self.stage = stage
        self.mode = stage

        self.args = args

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        rgbpath = self.rgb_list[idx]
        skepath = self.ske_list[idx]

        label = self.labels[idx]

        video = np.zeros([1])
        skeleton = np.zeros([1])

        if self.args.modality == "rgb" or self.args.modality == "both":
            video = load_video(rgbpath)
        if self.args.modality == "skeleton" or self.args.modality == "both":
            skeleton = get_3D_skeleton(skepath)

        video, skeleton = self.video_transform(self.args, video, skeleton)

        sample = {'rgb': video, 'ske': skeleton, 'label': label - 1}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def video_transform(self, args, np_clip, np_skeleton):
        if args.modality == "rgb" or args.modality == "both":
            # Div by 255
            np_clip /= 255.

            # Normalization
            np_clip -= np.asarray([0.485, 0.456, 0.406]).reshape(1, 1, 3)  # mean
            np_clip /= np.asarray([0.229, 0.224, 0.225]).reshape(1, 1, 3)  # std

        if args.modality == "skeleton" or args.modality == "both":
            # Take joint 2 of first person as origins for each person
            if args.no_norm == False:
                origin = np_skeleton[:, :, 1, 0]
                np_skeleton = np_skeleton - origin[:, :, None, None]

        return np_clip, np_skeleton


# %%
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", action="store",
                        dest="folder",
                        help="Path to the data",
                        default="NTU")
    parser.add_argument('--outputdir', type=str, help='output base dir', default='checkpoints/')
    parser.add_argument('--datadir', type=str, help='data directory', default='NTU')
    parser.add_argument("--j", action="store", default=12, dest="num_workers", type=int,
                        help="Num of workers for dataset preprocessing ")

    parser.add_argument("--vid_dim", action="store", default=256, dest="vid_dim",
                        help="frame side dimension (square image assumed) ")
    parser.add_argument("--vid_fr", action="store", default=30, dest="vi_fr", help="video frame rate")
    parser.add_argument("--vid_len", action="store", default=(8, 32), dest="vid_len", type=int, help="length of video")
    parser.add_argument('--modality', type=str, help='modality: rgb, skeleton, both', default='rgb')
    parser.add_argument("--hp", action="store_true", default=False, dest="hp", help="random search on hp")
    parser.add_argument("--no_norm", action="store_true", default=False, dest="no_norm",
                        help="Not normalizing the skeleton")

    parser.add_argument('--num_classes', type=int, help='output dimension', default=60)
    parser.add_argument('--batchsize', type=int, help='batch size', default=8)
    parser.add_argument("--clip", action="store", default=None, dest="clip", type=float,
                        help="if using gradient clipping")
    parser.add_argument("--lr", action="store", default=0.001, dest="learning_rate", type=float,
                        help="initial learning rate")
    parser.add_argument("--lr_decay", action="store_true", default=False, dest="lr_decay",
                        help="learning rate exponential decay")
    parser.add_argument("--drpt", action="store", default=0.5, dest="drpt", type=float, help="dropout")
    parser.add_argument('--epochs', type=int, help='training epochs', default=10)

    args = parser.parse_args()
    import torchvision.transforms as transforms

    transformer = transforms.Compose([NormalizeLen(), ToTensor()])
    train_transformer = transforms.Compose([NormalizeLen(), ToTensor()])
    dataset = NTU(args.folder, train_transformer, 'train', 32, args=args)
    iterator = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batchsize, shuffle=True,
                                           num_workers=args.num_workers)

    for batch in iterator:
        # print(batch["label"])
        print("ske", batch['ske'].shape, ", rgb", batch['rgb'].shape, ", label", batch['label'].shape)

        # print(batch["ske"])
        # check_skel(batch["ske"])
