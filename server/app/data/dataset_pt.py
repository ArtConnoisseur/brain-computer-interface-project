# =============================================

# Copyright (c) 2026 Rudraksh Srivastava
# SPDX-License-Identifier: AGPL 3.0

# This file is part of Rudraksh Srivastava's
# Final Year Project at the University
# of Edinburgh.

# This file contains the code for defining 
# pytorch Dataset and DataLoader objects 
# for the dataset post transformation in
# this study.

# =============================================

import mne 
import torch 

from torch.utils.data import Dataset, DataLoader

DATA_PATH = r"/dataset/processed"

def load_data(patient, trial, verbose=False):
    if trial in [1, 2]:
        print("Innvalid Trial number. Cannot be 1 or 2")
        return 
        
    complete_path = rf"{DATA_PATH}/S{patient:03}/S{patient:03}R{trial:02}_epo.fif"
    epoch = mne.read_epochs(complete_path, verbose=verbose)
    return epoch

# The above function has been deprecated 

class EEGDataset(Dataset):
    def __init__(
        self, 
        root_dir=r"/kaggle/input/datasets/artconnoisseur/processed-mi-dataset-for-fyp/processed", 
        patient_range=(1, 110),
        trial_range=(3, 13)
    ):
        self.root_dir = root_dir
        self.label_map = { idx + 1 : idx for idx in range(5) }
        self.all_patient_files = [
            self._get_patient_filepath(patient, trial)
            for patient in range(*patient_range)
            for trial in range(*trial_range)
        ]
    
    def __len__(self):
        return len(self.all_patient_files)

    def __getitem__(self, idx):
        """
        Get individual patient epoch, convert to 
        tensor and split into X and y. 

        Parameters:
            idx (int): Index of the file requested

        Returns:
            (tensor) X, y 
        """
        epochs = mne.read_epochs(self.all_patient_files[idx], preload=True, verbose=False)
        epochs.resample(128, npad="auto", verbose=False)

        X = torch.tensor(epochs.get_data(), dtype=torch.float32)
        y = torch.tensor([self.label_map[e] for e in epochs.events[:, 2]])

        return X, y

    def _get_patient_filepath(self, patient, trial):
        return rf"{DATA_PATH}/S{patient:03}/S{patient:03}R{trial:02}_epo.fif"

class EEGDatasetMultimodal(Dataset):
    """
    This class is used to get data for the multimodal 
    model 
    """
    def __init__(
        self, 
        root_dir=r"/kaggle/input/datasets/artconnoisseur/processed-mi-dataset-for-fyp/processed", 
        patient_range=(1, 110),
        trial_range=(3, 13),
        normalisation=True,
        wavelet=False 
    ):
        self.root_dir = root_dir
        self.label_map = { idx + 1 : idx for idx in range(5) }
        self.all_patient_files = [
            self._get_patient_filepath(patient, trial)
            for patient in range(*patient_range)
            for trial in range(*trial_range)
        ]
        self.normalise = normalisation
        self.wavelet = wavelet
    
    def __len__(self):
        return len(self.all_patient_files)

    def __getitem__(self, idx):
        """
        Get individual patient epoch, convert to 
        tensor and split into X and y. 

        Parameters:
            idx (int): Index of the file requested

        Returns:
            (tensor) X, y 
        """
        wavelet_transform = mne.time_frequency.tfr_morlet 
        fft = lambda epochs: mne.time_frequency.psd_array_welch(epochs.get_data(), sfreq=epochs.info["sfreq"])[0]

        epochs = mne.read_epochs(self.all_patient_files[idx], preload=True, verbose=False)
        epochs.resample(128, npad="auto", verbose=False)
        freq_bins = np.arange(0.5, 40, 2)

        X_time = torch.tensor(epochs.get_data(), dtype=torch.float32)
        X_image = torch.tensor(epochs.compute_tfr(method="morlet", freqs=freq_bins/2, n_cycles=freq_bins).data, dtype=torch.float32)
        # Mean Pool to collapse channels to form an image
        X_image = X_image.mean(dim=1)
        X_freq = torch.tensor(fft(epochs), dtype=torch.float32)

        y = torch.tensor([self.label_map[e] for e in epochs.events[:, 2]])

        def normalise(X):
            mean = X.mean() 
            std = X.std() 
            
            # Compute z scores 
            X = (X - mean)/std 
            
            return X

        X_time = normalise(X_time)
        X_freq = normalise(X_freq)
        X_image = normalise(X_image)
        

        return X_time, X_image, X_freq, y

    def _get_patient_filepath(self, patient, trial):
        return rf"{self.root_dir}/S{patient:03}/S{patient:03}R{trial:02}_epo.fif"