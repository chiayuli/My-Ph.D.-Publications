import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import torch

class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.feat_A = os.path.join(opt.dataroot, opt.phase + 'A/feats.scp')
        self.feat_B = os.path.join(opt.dataroot, opt.phase + 'B/feats.scp')

        self.A_dataset, self.A_uttid = make_dataset(self.feat_A)# load feat from '/path/to/data/trainA'
        self.B_dataset, self.B_uttid = make_dataset(self.feat_B)# load feat from '/path/to/data/trainB'
        self.A_size = len(self.A_dataset)  # get the size of dataset A
        self.B_size = len(self.B_dataset)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        #self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        #self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_data = self.A_dataset[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_data = self.B_dataset[index_B]
        A_uttid = self.A_uttid[index % self.A_size]
        B_uttid = self.B_uttid[index_B]
        # apply image transformation
        #A = self.transform_A(A_data)
        #B = self.transform_B(B_data)
        A = torch.FloatTensor(A_data)
        B = torch.FloatTensor(B_data)

        return {'A': A, 'B': B, 'A_uttid': A_uttid, 'B_uttid': B_uttid}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
