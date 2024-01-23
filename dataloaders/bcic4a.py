import numpy as np
from glob import glob

import torch
from torch.utils.data import Dataset
from dataloaders.eeg_transforms import ToTensor


class BCICompet2aIV(Dataset):
    def __init__(self, args):
        self.toTensor = ToTensor()
        self.args = args

        self.data, self.label = self.get_numpy_brain_data()

        print(self.data.shape, self.label.shape)

        self.data = torch.from_numpy(self.data).type(torch.FloatTensor)
        self.label = torch.from_numpy(self.label).type(torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

    def check_subject_num(self, i):
        if (i + 1) == self.args.subject_num:
            return True
        else:
            return False

    def get_numpy_brain_data(self):
        if self.args.domain_type == "source":
            filelist = sorted(glob(f"{self.args.BASE_PATH}/*T_X.npy"))
            label_filelist = sorted(glob(f"{self.args.BASE_PATH}/*T_Y.npy"))
        else:
            filelist = sorted(glob(f"{self.args.BASE_PATH}/*X.npy"))
            label_filelist = sorted(glob(f"{self.args.BASE_PATH}/*Y.npy"))

        if self.args.domain_type == "source":
            filelist_arr, label_filelist_arr = [], []

            for i, (file, label) in enumerate(zip(filelist, label_filelist)):
                if self.check_subject_num(i):
                    continue
                print("source", file, label)
                filelist_arr.append(np.load(file))
                label_filelist_arr.append(np.load(label))

            filelist = np.concatenate((filelist_arr))
            label_filelist = np.concatenate((label_filelist_arr))

        elif self.args.domain_type == "target":
            idx = self.args.subject_num * 2 - 1
            print("target", filelist[idx], label_filelist[idx])
            filelist = np.load(filelist[idx])
            label_filelist = np.load(label_filelist[idx])

        elif self.args.domain_type == "test":
            idx = self.args.subject_num * 2 - 2
            print("test", filelist[idx], label_filelist[idx])
            filelist = np.load(filelist[idx])
            label_filelist = np.load(label_filelist[idx])

        data = filelist
        label = label_filelist

        return data, label

    def get_numpy_brain_data_dependent(self):
        if self.args.train_type == "training":
            data = f"{self.args.BASE_PATH}/A0{self.args.subject_num}T_X.npy"
            label = f"{self.args.BASE_PATH}/A0{self.args.subject_num}T_Y.npy"
        elif self.args.train_type == "test":
            data = f"{self.args.BASE_PATH}/A0{self.args.subject_num}E_X.npy"
            label = f"{self.args.BASE_PATH}/A0{self.args.subject_num}E_Y.npy"

        if self.args.train_type == "training":
            print("training", data, label)
            data = np.load(data)
            label = np.load(label)

        elif self.args.train_type == "test":
            print("test", data, label)
            data = np.load(data)
            label = np.load(label)

        return data, label
