"""A modified image folder class

We modify the official PyTorch image folder (https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py)
so that this class can load images from both current directory and its subdirectories.
"""

import torch.utils.data as data
import torch.nn.functional as F
import torch
import torchaudio
torchaudio.set_audio_backend("sox_io")
from torchaudio.kaldi_io import read_mat_scp
torchaudio.set_audio_backend("sox_io")
from PIL import Image
import os

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(feat_file, max_dataset_size=float("inf")):
    feats = []
    uttids = []
    max_len = 0
    context=5
    i=0
    assert os.path.isfile(feat_file), '%s is not a valid feature file' % feat_file
    for key, mat in read_mat_scp(feat_file):
        zero_Tensor=torch.zeros((5,40))
        pad_mat=torch.cat((zero_Tensor,mat), 0)
        pad_mat=torch.cat((pad_mat,zero_Tensor), 0)
        for i in range(context,(int(mat.size()[0])+context)):
            uttids.append(key)
            start=i-context
            end=i+context+1
            new_mat=torch.FloatTensor(pad_mat[start:end]).unsqueeze(0)
            feats.append(new_mat)
    return feats, uttids


def make_dataset_aligned(feat_A_file, feat_B_file, max_dataset_size=float("inf")):
    feat_A = {}
    feat_B = {}
    feat_AB = {}
    max_len = 0
    context=5
    i=0
    assert os.path.isfile(feat_file), '%s is not a valid feature file' % feat_A_file
    assert os.path.isfile(feat_file), '%s is not a valid feature file' % feat_B_file
    ## process feat_A
    for key, mat in read_mat_scp(feat_A_file):
        feat_per_utt=[]
        zero_Tensor=torch.zeros((5,40))
        pad_mat=torch.cat((zero_Tensor,mat), 0)
        pad_mat=torch.cat((pad_mat,zero_Tensor), 0)
        for i in range(context,(int(mat.size()[0])+context)):
            start=i-context
            end=i+context+1
            new_mat=torch.FloatTensor(pad_mat[start:end]).unsqueeze(0)
            feat_per_utt.append(new_mat)
        feat_A.update({str(key):feat_per_utt})

    ## process feat_B
    for key, mat in read_mat_scp(feat_B_file):
        feat_per_utt=[]
        zero_Tensor=torch.zeros((5,40))
        pad_mat=torch.cat((zero_Tensor,mat), 0)
        pad_mat=torch.cat((pad_mat,zero_Tensor), 0)
        for i in range(context,(int(mat.size()[0])+context)):
            start=i-context
            end=i+context+1
            new_mat=torch.FloatTensor(pad_mat[start:end]).unsqueeze(0)
            feat_per_utt.append(new_mat)
        feat_B.update({str(key):feat_per_utt})

     ## conbime feat_A and feat_B
     #for key, matlist in feat_A:
     #    combinelist=[]
     #    if key in feat_B:
     #        for i in range(len(matlist)):
     #            combine=torch.cat((matlist[i], feat_B[key][i]), 0)
     #            print(combine.size())
     #            combinelist.append(combine)
     #        feat_AB.update({str(key):combinelist})
     #    else:
     #       print("There must be a bug...")

    return feat_AB

def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)
