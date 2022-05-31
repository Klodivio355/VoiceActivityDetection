import numpy as np
import torch
from torch.utils.data import Dataset


class VoiceActivityDetection(Dataset):
    """Voice Activity Detection dataset."""

    def __init__(self, option=None, transform=None, target_transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            option (string): train, valid or test.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.option = option
        with open('processed/' + self.option + '/audio/audio.npy', 'rb') as f:
            self.audios = np.load(f)
        with open('processed/' + self.option + '/labels/labels.npy', 'rb') as f:
            self.labels = np.load(f)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.audios)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio = self.audios[idx]
        label = self.labels[idx]
        label = np.asarray(label, dtype='float64')

        if self.transform:
            audio = self.transform(audio)

        if self.target_transform:
            label = self.target_transform(label)

        return audio, label
