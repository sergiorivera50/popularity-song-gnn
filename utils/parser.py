import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import h5py
from math import floor
import mne
from mne.preprocessing import ICA, create_eog_epochs, create_ecg_epochs, corrmap
import warnings
from time import time
from math import ceil

warnings.filterwarnings("ignore")
mne.set_log_level("CRITICAL")
plt.rcParams["figure.figsize"] = (16, 8)


class FeatureExtractor:
    """
    For a given recording, extract all the corresponding features and add the track_id as row.idx
    """

    def __init__(self, video_list_path, features_file):
        self.df = pd.read_csv(features_file)
        vid_list = open(video_list_path).readlines()
        # str( • ) cast to string to match in the df column
        # •.split(" ")[1] remove the "file" at beginning
        # •.strip() removes \n
        # •[3:-5] removes 1_ and .mp4 from the string
        # •.split("_") divide the title
        # if idx % 2 even lines are obsolete
        self.vid_list = [str(l.split(" ")[1].strip()[3:-5].split("_"))
                         for idx, l in enumerate(vid_list)
                         if idx % 2]

    def extract_track_features(self, idx):
        return self.df.loc[self.df['title'] == self.vid_list[idx]].to_numpy()[0, 1:].astype(np.float32)

    def extract_record_features(self):
        features = []
        for idx in range(0, 20):
            features.append(self.extract_track_features(idx))
        return np.vstack(features)


class RecordingELT:
    """
    Extract Load Transform for the batch of tracks of a whole recording.
    """

    def __init__(self, fname):
        assert fname != "" and fname is not None
        self.fname = fname
        self.data = self.load_data()
        # instead of giving a track_id == None, we set the initial track as a default value
        self.elt_track = TrackELT(self.data, 0)
        self.elt_track.fname = fname  # handling an internal error

    def load_data(self):
        assert self.fname != "" and self.fname is not None
        f = h5py.File(self.fname, 'r')
        data = f.get('y')
        data = np.array(data)
        # col 0 is the time, sampled at 250 samples per second
        # col -1 is all 11, we'll drop it
        # col -2 is the trigger
        data = data[:, 1:-1]
        data[:, -1] -= 1  # useful for sign changes
        return data

    def clean_single_track(self, track_id: int):
        """
        For a given EEG measurement as .mat file, clean the track_id-th track.
        :param track_id: integer between [0, 19].
        """

        self.elt_track.track_id = track_id
        return self.elt_track.clean_data(self.data)

    def clean_all_tracks(self):
        """
        Perform the cleaning operation over all tracks for a recording.
        After this operation tracks have to be aligned.
        """

        cleaned_tracks = []
        for track_id in range(0, 20):  # range in interval [0, 20) -> [0, 19]
            cleaned_tracks.append(self.clean_single_track(track_id))
        self.cleaned_tracks = cleaned_tracks
        return cleaned_tracks

    def align_cleaned_tracks(self):
        try:
            if self.cleaned_tracks is None or self.cleaned_tracks == []:
                self.clean_all_tracks()
        except:
            self.clean_all_tracks()

        min_len = min([track.shape[1] for track in self.cleaned_tracks])
        aligned_tracks = []
        for track in self.cleaned_tracks:
            diff = track.shape[1] - min_len
            # handle equal alignment
            if diff == 0:
                aligned_tracks.append(np.expand_dims(track, 0))
            else:
                a = b = diff / 2
                a = floor(a)
                b = ceil(b)
                # if you have an odd diff, trim a bonus value from the end
                if b == 0:
                    b += 1
                """
                if track[:, a:-b].shape[1] > min_len:
                    a += 1
                elif track[:, a:-b].shape[1] < min_len:
                    b += 1
                """
                aligned_tracks.append(np.expand_dims(track[:, a:-b], 0))
        self.aligned_tracks = np.vstack(aligned_tracks)
        return self.aligned_tracks


class TrackELT:
    """
    Extract Load Transform for a single track, for a single person
    """

    def __init__(self, track, track_id):
        self.n_channels = 32  # 32 channels
        self.s_freq = 250  # 250 sample per second
        self.ch_types = ['eeg'] * self.n_channels  # all eegs'
        self.track = track
        self.track_id = track_id
        self.start, self.stop = self.extract_trigger_intervals(track)
        assert self.track_id >= 0 and self.track_id < 20

    def extract_trigger_intervals(self, data):
        """
        Extract a list of tuples that contain respectively the start and the stop of the trigger. -- deprecated
        Extract the start & stop time intervals for a given track id in [0, 19].
        """

        assert self.track_id >= 0 and self.track_id < 20
        signchange = ((np.roll(data[:, -1], 1) - data[:, -1]) != 0).astype(int)
        trig = np.where(signchange == 1)[0]
        trig_coupled = np.array([(trig[i], trig[i + 1]) for i in range(0, len(trig) - 1, 2)])
        return trig_coupled[self.track_id][0], trig_coupled[self.track_id][1]

    def prepare_data(self, data):
        """
        data_trim: Signal in a single sample period (30s) as (n_samples x channels) ndarray.
        """

        montage = mne.channels.make_standard_montage('biosemi32')
        data_info = mne.create_info(ch_names=montage.ch_names,
                                    sfreq=self.s_freq,
                                    ch_types=self.ch_types,
                                    verbose=True)

        # data_mne_raw = mne.io.RawArray(data.T, data_info) # --probably obsolete
        # data_mne_raw._filenames = [self.fname.split('/')[-1]] #avoid annoying errors --probably obsolete

        data_filtered = mne.filter.filter_data(data.T,
                                               self.s_freq,
                                               l_freq=1.0, h_freq=None,
                                               picks=None)  # HPF

        data_mne_filtered = mne.io.RawArray(data_filtered, data_info)
        data_mne_filtered._filenames = [self.fname.split('/')[-1]]  # avoid annoying errors

        new_names = dict(
            (ch_name,
             ch_name.rstrip('.').upper().replace('Z', 'z').replace('FP', 'Fp'))
            for ch_name in data_mne_filtered.ch_names)
        data_mne_filtered.rename_channels(new_names)

        data_mne_filtered.set_montage(montage)
        data_mne_filtered.set_eeg_reference(projection=True)  # needed for inverse modeling
        return data_mne_filtered

    def run_ica(self, data, method, n_components, fit_params=None):
        ica = ICA(n_components=n_components, method=method, fit_params=fit_params,
                  max_iter=120, random_state=0, verbose=False)
        t0 = time()
        ica.fit(data)
        fit_time = time() - t0
        title = ('ICA decomposition using %s (took %.1fs)' % (method, fit_time))
        return ica

    def clean_data(self, data, n_components=4, verbose=False):
        """
        DOC
        """

        t0 = time()
        # restrict the time serie domain to [start, stop)
        # trigger is useless from now on
        self.start, self.stop = self.extract_trigger_intervals(self.track)
        print("\t", self.start, self.stop, self.stop - self.start)
        data_trim = data[self.start:self.stop + 1, :-1]
        # NEEDED HPF filter for ICA
        data_mne_filtered = self.prepare_data(data_trim).filter(1.0, None)
        ica = self.run_ica(data_mne_filtered, 'fastica', n_components)
        data_copy = data_mne_filtered.copy()
        ica.apply(data_copy)
        if verbose:
            self.show_projection(data_mne_filtered, ica)
            plt.plot(data_trim[:, 0], label='original')
            plt.plot(data_copy.get_data()[0, :], label='clean')
            plt.legend()

        fit_time = time() - t0
        title = ('Cleaning the %d-th track took %.3fs)' % (self.track_id, fit_time))
        self.data_cleaned = data_copy.get_data()  # should take the array or the object here?!
        return data_copy.get_data()

    def show_projection(self, data, ica):
        ica.plot_components(ch_type='eeg')
        ica.plot_sources(data, show_scrollbars=False)
        ica.plot_overlay(data, exclude=[0, 1], picks='eeg')
        ica.plot_properties(data, verbose=False)


class ELT:
    def __init__(self, eeg_path="data/signals_anon", features_path="data/features", songs_lists="data/songs_lists"):
        self.eeg_path = eeg_path
        self.feat_path = features_path
        self.songs_lists = songs_lists
        self.eeg_files = [fname for fname in os.listdir(self.eeg_path) if fname.endswith(
            ".mat")]  # this will be useful when loading the songs' list files (for indexing)

    def load_recs(self):
        not_aligned_recs = []
        for fname in os.listdir(self.eeg_path):
            if not fname.endswith("mat"):
                continue
            print(fname)
            elt_rec = RecordingELT(self.eeg_path + "/" + fname)
            elt_rec.clean_all_tracks()
            not_aligned_recs.append(elt_rec.align_cleaned_tracks())
        # self.batch = np.vstack(aligned_tracks)
        aligned_recs = []
        min_len = min([s.shape[2] for s in not_aligned_recs])
        for rec in not_aligned_recs:
            diff = rec.shape[2] - min_len
            # handle equal alignment
            if diff == 0:
                aligned_recs.append(rec)
            else:
                a = b = diff / 2
                a = floor(a)
                b = ceil(b)

                if b == 0:
                    b += 1

                aligned_recs.append(rec[:, :, a:-b])
        return aligned_recs

    def load_features(self):
        try:
            assert len(self.eeg_files) > 0
        except:
            self.eeg_files = [fname for fname in os.listdir(self.eeg_path) if fname.endswith("mat")]

        features = []
        y = []
        print(self.eeg_files)
        for eeg_file in self.eeg_files:
            rec_idx = eeg_file.split("_")[1].split('.')[0]
            fe = FeatureExtractor("data/songs_lists/" + str(rec_idx) + "_video_list.txt", "data/features.csv")
            tmp_features = fe.extract_record_features()
            y.append(tmp_features[:, 0])
            features.append(tmp_features[:, 1:])
        return np.vstack(features), np.hstack(y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', default='data/signals_anon', help='path to the eeg directory')
    args = parser.parse_args()

    elt = ELT(args.path)
    e = elt.load_recs()
    eeg_array = np.vstack(e)
    X, y = elt.load_features()
    print(f"Shape: {eeg_array.shape}")
    print(f"X: {X.shape}")
    print(f"y: {y.shape}")

    np.save("./data/parsed.npy", {
        "eeg_array": eeg_array,
        "X": X,
        "y": y,
    })
