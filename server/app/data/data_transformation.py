# =============================================

# Copyright (c) 2026 Rudraksh Srivastava
# SPDX-License-Identifier: AGPL 3.0

# This file is part of Rudraksh Srivastava's
# Final Year Project at the University
# of Edinburgh.

# This file contains the code for data
# transformation, mainly converting the data
# from EDF+ files to HDF5 files, as well as
# making the annotations more specific for
# multihead classification of all actions.

# The code for preprocessing and dataloaders
# are maintained in their respective modules.

# =============================================

import os

import h5py
import mne
import numpy as np
from natsort import natsorted
from pyedflib import highlevel



class DataTransformer:
    """
    This class provides the methods to convert
    EDF+ files to HDF+ files.
    """

    def __init__(self, data_directory: str):
        """
        Initialises the class

        Args:
            data_directory (str): Path to the
                data_directory.

        Returns:
            None.
        """

        self.raw_data_directory = rf"{data_directory}/files"
        self.reannotation_directory = r"eeg-data/reannotated"
        self.patient_files = [
            file
            for file in natsorted(os.listdir(self.raw_data_directory))
            if file.startswith("S0") or file.startswith("S1")
        ]

        self.patient_edf_files = {
            patient_file: [
                file
                for file in natsorted(
                    os.listdir(f"{self.raw_data_directory}/{patient_file}")
                )
                if file.endswith(".edf")
            ]
            for patient_file in self.patient_files
        }

        print("Patient Files Loaded")

    def write_to_hdf5(self):
        pass

    def compute_new_annotations(self, patient: str = "S001"):
        """
        This function computes the new annotations
        and stores it in a new method (while also
        returning it)

        Args:
            paitent (str): Patient ID String
                Default: S0001.

        Returns:
            (dict): Same format as patient_edf_files
                with correct annotations.
        """

        edf_files = self.patient_edf_files[patient]

        # Overall Mapping:
        #
        # integer_annotation_mapping = {
        #     "0": "No event occuring.",
        #     "1": "Closing Left Fist.",
        #     "2": "Closing Right Fist.",
        #     "3": "Closing Both Fists.",
        #     "4": "Closing Both Feet.",
        # }

        # Define re-annotation maps:
        left_right_annotation = {
            "T1": 1,
            "T0": 0,
            "T2": 2,
        }

        fist_feet_annotation = {
            "T0": 0,
            "T1": 3,
            "T2": 4,
        }

        for file in edf_files:
            # Get the task number
            task_num = int(file.rstrip(".edf")[-2:])
            # Read the files
            raw = mne.io.read_raw_edf(f"{self.raw_data_directory}/{patient}/{file}")

            match task_num:
                case 1 | 2:
                    print("Baseline task number.")
                case 3 | 4 | 7 | 8 | 11 | 12:
                    raw.annotations.rename(left_right_annotation)
                    print("Left/Right task number.")
                case 5 | 6 | 9 | 10 | 13 | 14:
                    raw.annotations.rename(fist_feet_annotation)
                    print("Fist/Feet task number.")

            try:
                os.mkdir(f"{self.reannotation_directory}/{patient}")
            except FileExistsError:
                pass

            raw.save(
                f"{self.reannotation_directory}/{patient}/{file.rstrip('.edf')}_raw.fif",
                overwrite=True,
            )

    def reannotate_all_eeg_data(self):
        """
        ToDo: Add docstrings
        """

        for patient in self.patient_files[1:]:
            self.compute_new_annotations(patient=patient)

if __name__ == "__main__":
    dt = DataTransformer(data_directory="eeg-data")
    # dt.compute_new_annotations()
    dt.reannotate_all_eeg_data()
