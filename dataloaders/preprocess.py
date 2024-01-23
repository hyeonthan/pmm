import os
import numpy as np
from glob import glob
from tqdm import tqdm
import mne
import scipy

from braindecode.datautil.preprocess import exponential_moving_standardize


def preprocess_of_brain_data(base_path, save_path, down_sampling):
    """
    * 769: Left
    * 770: Right
    * 771: foot
    * 772: tongue
    """
    sfreq = 250
    tmin_value = -2.0
    tmax_value = 4.0
    time_period = abs(tmax_value) + abs(tmin_value)

    filelist = sorted(glob(f"{base_path}/bcic4a/*.gdf"))
    label_filelist = sorted(glob(f"{base_path}/bcic4a_label/*.mat"))

    data, label = [], []

    for idx, filename in enumerate(tqdm(filelist)):
        print(f"LOG >>> Filename: {filename}")

        raw = mne.io.read_raw_gdf(filename, preload=True)
        events, annot = mne.events_from_annotations(raw)

        raw.filter(0.5, 38.0, fir_design="firwin")
        raw.info["bads"] += ["EOG-left", "EOG-central", "EOG-right"]

        picks = mne.pick_types(
            raw.info, meg=False, eeg=True, eog=False, stim=False, exclude="bads"
        )

        tmin, tmax = tmin_value, (sfreq * tmax_value - 1) / sfreq

        file_type = filename[-5]
        if file_type == "E":
            event_id = dict({"783": 7})
        elif file_type == "T":
            event_id = (
                dict({"769": 7, "770": 8, "771": 9, "772": 10})
                if idx != 7
                else dict({"769": 5, "770": 6, "771": 7, "772": 8})
            )

        epochs = mne.Epochs(
            raw,
            events,
            event_id,
            tmin,
            tmax,
            proj=True,
            picks=picks,
            baseline=None,
            preload=True,
        )

        if down_sampling != 0:
            epochs = epochs.resample(down_sampling)
        fs = epochs.info["sfreq"]

        epochs_data = epochs.get_data() * 1e6
        splited_data = []

        for epoch in epochs_data:
            normalized_data = exponential_moving_standardize(
                epoch, init_block_size=int(epochs.info["sfreq"] * time_period)
            )
            splited_data.append(normalized_data)

        splited_data = np.stack(splited_data)
        splited_data = splited_data[:, np.newaxis, ...]

        label_list = scipy.io.loadmat(label_filelist[idx])["classlabel"].reshape(-1) - 1

        data = splited_data
        label = label_list

        data_filename = os.path.splitext(os.path.basename(filename))[0]
        label_filename = os.path.splitext(os.path.basename(label_filelist[idx]))[0]
        np.save(os.path.join(save_path, f"{data_filename}_X.npy"), data)
        np.save(os.path.join(save_path, f"{label_filename}_Y.npy"), label)


if __name__ == "__main__":
    base_path = "datasets/BCI_Competition_IV/2a"
    save_path = "datasets/preprocess_data_250/2a_6sd_front"
    down_sampling = 0
    preprocess_of_brain_data(base_path, save_path, down_sampling)
