from __future__ import print_function
from torch.utils import data
import os
import numpy as np
# from scipy.ndimage import imread
from matplotlib.pyplot import imread
import random
import config
from torchvision import transforms


class BDIDerivedFolder(data.Dataset):
    def __init__(self, folder, submedian, flip_aug, transform=None, label=None):
        self.folder = folder
        self.flip_aug = flip_aug
        self.transform = transform
        
        # check folders
        if not submedian:
            self.faces_folder = os.path.join(folder)
        else:
            self.faces_folder = os.path.join(folder, "faces-sub-median")
            
        if not os.path.exists(self.faces_folder):
            raise RuntimeError("Folder {} is not a valid experiment folder.".format(self.folder))
        
        # calculate number of samples (len() of the folder)
        
        self.length = (len(os.listdir(self.faces_folder)) - config.LEN_SAMPLE) // config.NONOVERLAP + 1
        self.label = self.get_label()

    def __getitem__(self, index):
        if self.flip_aug and random.random() < 0.5:
            flip = True
        else:
            flip = False
        frames = [self._getframe(idx, flip=flip) for idx in range(index * config.NONOVERLAP, index * config.NONOVERLAP + config.LEN_SAMPLE)]
        
        # combine faces  into faces_all, shape: T x 3 x 90 x 90
        faces_all = np.array([frame[0] for frame in frames])
        
        item = [faces_all, [frames[0][1]]]
        return item

    def __len__(self):
        return self.length

    def _getframe(self, index, flip):
        # load face image, shape: (#images x F_H) x F_W
        face_file = os.path.join(self.faces_folder, str(index+1).zfill(8) + ".png_Aligned.png")
        
        # TODO: many missing files
        if not os.path.exists(face_file):
            face_file = os.path.join(self.faces_folder, str(1).zfill(8) + ".png_Aligned.png")

        face_img = imread(face_file)
        
        height, width, channels = face_img.shape
        
        if not (height==config.F_H and  width==config.F_W and channels==config.C):
            raise RuntimeError("File {} is invalid with dimnesion of {}, {}, {}"
                               .format(face_file,height, width, channels))
            
        if self.transform is not None:
            face_img = self.transform(face_img)
            
        # data augmentation
        if flip:
            face_img = self._face_horizontal_flip(face_img)

        return face_img, self.label

    def get_label(self):
        """ Return the label of the data sample based on the folder name"""
        # Get label ID
        path = os.path.normpath(self.folder)
        parts = path.split(os.sep)
        return config.CLASSES[parts[-2]]
    
    @staticmethod
    def _face_horizontal_flip(img):
        return img
        """ Horizontally flip the given image """
        # flip all images horizontally
        img = np.flip(img, 1)
        # exchange left and right shoulders
        img[list(range(1*config.F_H, 2*config.F_H))+list(range(4*config.F_H, 5*config.F_H))]\
            = img[list(range(4*config.F_H, 5*config.F_H))+list(range(1*config.F_H, 2*config.F_H))]
        # exchange left and right elbows
        img[list(range(2*config.F_H, 3*config.F_H))+list(range(5*config.F_H, 6*config.F_H))]\
            = img[list(range(5*config.F_H, 6*config.F_H))+list(range(2*config.F_H, 3*config.F_H))]
        # exchange left and right wrists
        img[list(range(3*config.F_H, 4*config.F_H))+list(range(6*config.F_H, 7*config.F_H))]\
            = img[list(range(6*config.F_H, 7*config.F_H))+list(range(3*config.F_H, 4*config.F_H))]
        # exchange left and right hips
        img[list(range(7*config.F_H, 8*config.F_H))+list(range(10*config.F_H, 11*config.F_H))]\
            = img[list(range(10*config.F_H, 11*config.F_H))+list(range(7*config.F_H, 8*config.F_H))]
        # exchange left and right knees
        img[list(range(8*config.F_H, 9*config.F_H))+list(range(11*config.F_H, 12*config.F_H))]\
            = img[list(range(11*config.F_H, 12*config.F_H))+list(range(8*config.F_H, 9*config.F_H))]
        # exchange left and right ankles
        img[list(range(9*config.F_H, 10*config.F_H))+list(range(12*config.F_H, 13*config.F_H))]\
            = img[list(range(12*config.F_H, 13*config.F_H))+list(range(9*config.F_H, 10*config.F_H))]
        return img


class BDIDerivedDataset(data.Dataset):
    def __init__(self, folders, submedian, flip, return_idx=False):
        self.folders = folders
        self.derived_folders = [BDIDerivedFolder(f, submedian, flip)
                               for f in folders]
        
        self.len_folders = [len(f) for f in self.derived_folders]
        self.length = sum(self.len_folders)
        self.lookup_table = self._build_lookup_table()
        self.return_idx = return_idx

    def __getitem__(self, index):
        idx1, idx2 = self.lookup_table[index]
        if not self.return_idx:
            return self.derived_folders[idx1][idx2]
        else:
            return self.derived_folders[idx1][idx2], self.folders[idx1].split('/')[-1], idx2

    def __len__(self):
        return self.length

    def _build_lookup_table(self):
        lvl1, lvl2 = (), ()
        for idx, derived_folder in enumerate(self.derived_folders):
            lvl1 += (idx,) * len(derived_folder)
            lvl2 += tuple(range(len(derived_folder)))
        table = [(l1, l2) for l1, l2 in zip(lvl1, lvl2)]
        return table
