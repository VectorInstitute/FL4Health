import argparse
import collections
import copy
import enum
import json
import os
import pickle
import random
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SubsetRandomSampler


def add_fit_args(parser):
    parser.add_argument("--data_path", type=str, default="data_storage", help="Data path")
    args = parser.parse_args()
    return args


args = add_fit_args(argparse.ArgumentParser())


def depickle(filepath):
    with open(filepath, mode="rb") as f:
        return pickle.load(f)


class ImputationTypes(enum.Enum):
    ZERO = enum.auto()
    BFILL = enum.auto()
    FFILL = enum.auto()
    LINEAR = enum.auto()


def bfill(df):
    return df.fillna(method="bfill")


def ffill(df):
    return df.fillna(method="ffill")


def zero(df):
    return df.fillna(value=0)


def interpolate(df):
    return df.interpolate(method="linear")


IMPUTATION_FUNCTIONS = {
    ImputationTypes.BFILL: bfill,
    ImputationTypes.FFILL: ffill,
    ImputationTypes.ZERO: zero,
    ImputationTypes.LINEAR: interpolate,
}

EVAL_MODES = ("all_time", "first_24", "extend_till_discharge")


def tokenize_notes(notes, tokenizer):
    notes = None
    return notes


############################################################################


class PatientDataset(Dataset):
    def __init__(
        self,
        # These params are constructed explicitly by the dataset maker.
        rolling_ftseq: pd.DataFrame,
        rolling_tasks_continuous: pd.DataFrame,
        rolling_tasks_to_embed: pd.DataFrame,
        static_tasks_continuous: pd.DataFrame,
        static_tasks_to_embed: pd.DataFrame,
        ts_continuous: pd.DataFrame,
        ts_to_embed: pd.DataFrame,
        statics_continuous: pd.DataFrame,
        statics_to_embed: pd.DataFrame,
        notes: pd.DataFrame,
        max_hours_map: Dict[int, int],
        seed: int = 1,
        vocab: Dict = {},
        all_vocabs: Dict = {},
        max_time_since_measured: int = 8,
        max_seq_len: int = 480,  # 20 days.
        min_seq_len: int = 24,
        # min_seq_len:   int = 25,
        imputation_method: ImputationTypes = ImputationTypes.LINEAR,
        sequence_length: int = 25,  # None
        # sequence_length = None,
        do_all_timepoints: bool = False,
        num_random_endpoints: int = 0,
        extend_till_discharge: bool = False,
        imputation_mask_rate: float = 0,
    ):

        self.all_vocabs = all_vocabs

        ts_to_embed = ts_to_embed.astype(np.int32)
        self.ts_continuous_cols = ts_continuous.columns

        ts_to_embed_vocab = {k: v for k, v in vocab.items() if k in ("DNR Ordered", "Comfort Measures Ordered")}
        for c in ts_to_embed.columns:
            if type(c) is tuple and c[1] == "time_since_measured":
                ts_to_embed_vocab[c] = ["%d hours" % x for x in range(max_time_since_measured + 1)]

        # one hot encode using pandas
        ts_to_embed_one_hot = pd.get_dummies(ts_to_embed.astype(str))

        ts = pd.concat((ts_continuous, ts_to_embed_one_hot), axis=1)

        statics_to_embed = statics_to_embed.astype(np.int32)
        statics_to_embed_one_hot = pd.get_dummies(statics_to_embed.astype(str))
        # make assertions about statics_to_embed_one_hot

        statics = pd.concat((statics_continuous, statics_to_embed_one_hot), axis=1)

        # Do stuff...
        # TODO(mmd): consistent naming.
        dfs = [
            ("rolling_ftseq", rolling_ftseq),
            ("rolling_tasks_binary_multilabel", rolling_tasks_continuous),  ## mortality
            ("rolling_tasks_multiclass", rolling_tasks_to_embed),  ## discharge
            ("static_tasks_binary_multilabel", static_tasks_continuous),  # long length
            ("static_tasks_multiclass", static_tasks_to_embed),  # Final Acuity Outcome
            ("ts", ts),
            ("statics", statics),
        ]

        dfs = [(k, df) for k, df in dfs if df is not None and df.shape[1] > 0]

        # We use this awkward, roundabout construction to preserve key ordering.
        self.dfs = {k: df for k, df in dfs}
        self.keys = [k for k, df in dfs]

        # TODO(mmd): This is entirely antithetical to the whole design principle here... Fix it.
        self.multiclass_sizes = {}
        for k in ("rolling_tasks_multiclass", "static_tasks_multiclass"):
            df = self.dfs[k]
            for c in df.columns:
                assert c not in self.multiclass_sizes, "Collision!"
                self.multiclass_sizes[c] = df[c].max()

        self.dfs["next_timepoint"] = self.dfs["ts"].copy()
        # drop cols that have measured and time_since
        self.keys.append("next_timepoint")

        self.dfs["next_timepoint_was_measured"] = (ts_to_embed == 0).astype(float)
        if "DNR Ordered" in self.dfs["next_timepoint_was_measured"].columns:
            self.dfs["next_timepoint_was_measured"].drop(
                columns=["DNR Ordered", "Comfort Measures Ordered"], inplace=True
            )
        self.keys.append("next_timepoint_was_measured")

        self.keys.append("ts_mask")

        self.orig_max_seq_len = min(max_seq_len, max(max_hours_map.values()))
        self.orig_min_seq_len = min_seq_len
        self.orig_subjects = sorted(subj for subj, hrs_in in max_hours_map.items())
        self.orig_max_hours = [max_hours_map[subj] for subj in self.orig_subjects]

        self.subjects = [s for s in self.orig_subjects]

        self.seed = seed
        self.impute_fn = IMPUTATION_FUNCTIONS[imputation_method]
        self.max_seq_len = min(max_seq_len, max(max_hours_map.values()))

        self.min_seq_len = min_seq_len

        self.do_all_timepoints = do_all_timepoints
        self.num_random_endpoints = num_random_endpoints
        self.extend_till_discharge = extend_till_discharge
        self.imputation_mask_rate = imputation_mask_rate

        self.reset_sequence_len(sequence_length)

        self.binary_multilabel_task_concat_order = [
            "rolling_tasks_binary_multilabel",
            "static_tasks_binary_multilabel",
        ]

        self.train_tune_test = "train"
        self.epoch = 0
        self.save_path = f"{args.data_path}/eicu-2.0/federated_preprocessed_data/cached_data"
        path = Path(self.save_path)
        path.mkdir(parents=True, exist_ok=True)

    def reset_sequence_len(self, new_sequence_len, reset_index=True):
        self.sequence_len = new_sequence_len
        if self.sequence_len:
            assert not self.do_all_timepoints
            assert not self.extend_till_discharge
            assert self.sequence_len > self.orig_min_seq_len and self.sequence_len < self.orig_max_seq_len
            self.min_seq_len = self.sequence_len - 1
            self.max_seq_len = self.sequence_len

        max_hours_map = {subj: hrs_in for subj, hrs_in in zip(self.orig_subjects, self.orig_max_hours)}
        self.subjects = sorted(subj for subj, hrs_in in max_hours_map.items() if hrs_in > self.min_seq_len)
        self.max_hours = [max_hours_map[subj] for subj in self.subjects]

        if reset_index:
            self.reset_index()

    def reset_index(self):
        self.index = []
        if self.do_all_timepoints:
            for subject, max_hour in zip(self.subjects, self.max_hours):
                self.index.extend([(subject, hr) for hr in range(self.orig_min_seq_len, max_hour)])
        elif self.num_random_endpoints:  # 10 일 때 확인할 필요 있음
            for subject, max_hour in zip(self.subjects, self.max_hours):
                possible_hours = list(range(self.orig_min_seq_len, max_hour))
                if len(possible_hours) >= self.num_random_endpoints:
                    random_endpoints = np.random.choice(possible_hours, self.num_random_endpoints, replace=False)
                else:
                    random_endpoints = possible_hours

                self.index.extend([(subject, hr) for hr in random_endpoints])
        elif self.extend_till_discharge:
            self.index.extend(list(zip(self.subjects, self.max_hours)))
        else:
            self.index = self.subjects

    def set_to_eval_mode(self, eval_mode, num_random_endpoints=1):
        """
        Sets the dataset to operate in one of the foundational evaluation modes--either, first 24 hours,
        all time (epitomized through N random selections per patients, usually 1 for training and 10 for
        eval), or extend_till_discharge.
        """
        # Only used for tracking
        self.eval_mode = eval_mode

        # Constant changes regardless of eval_mode
        self.do_all_timepoints = False
        self.num_random_endpoints = 0
        self.sequence_len = None
        self.extend_till_discharge = False

        if eval_mode == "all_time":
            self.num_random_endpoints = num_random_endpoints
        elif eval_mode == "first_24":
            self.sequence_len = 25  # One extra to get full first day
        elif eval_mode == "extend_till_discharge":
            self.extend_till_discharge = True

        self.reset_index()

    def set_binary_multilabel_keys(self):
        self.binary_multilabel_keys = self.get_binary_multilabel_keys()

    def get_binary_multilabel_keys(self):
        if hasattr(self, "binary_multilabel_keys"):
            return self.binary_multilabel_keys

        out = []
        for key in self.binary_multilabel_task_concat_order:
            out.extend(list(self.dfs[key].columns))
        return out

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        """
        Returns:
            tensors (dict)
                'rolling_ftseq', [batch_size, 119]
                'ts', [batch_size, 239, 184]
                'statics', [batch_size, 54]
                'next_timepoint', [batch_size, 56]
                'next_timepoint_was_measured', [batch_size, 56]
                'disch_24h', [batch_size, 1]
                'disch_48h', [batch_size, 1]
                'Final Acuity Outcome', [batch_size, 1]
                'ts_mask', [batch_size, 239]
                'tasks_binary_multilabel', [batch_size, 7]
        """
        # We'll use these for a bit of special processing surrounding our masked imputation task, so we
        # define them now.
        ts_vals_key, ts_is_measured_key, imputation_mask_key = "ts_vals", "ts_is_measured", "ts_mask"

        # Icustay id is always first.
        idx = self.index[item]

        if type(idx) is tuple:
            icustay_id, end_time = idx
            start_time = max(end_time - self.max_seq_len, 0)
            seq_len = end_time - start_time
        else:
            icustay_id = idx
            if self.sequence_len:
                end_time = self.sequence_len
                start_time = max(end_time - self.max_seq_len, 0)
                seq_len = end_time - start_time
            else:
                max_seq_len = min(self.max_hours[item], self.max_seq_len)
                end_time = random.randint(self.min_seq_len, self.max_hours[item])  # the end time for this patient
                start_time = max(end_time - max_seq_len, 0)  # the start time corresponding to the random_end_time
                seq_len = end_time - start_time

        # collect the indices for the patient
        idxs = {k: (df.index.get_level_values("icustay_id") == icustay_id) for k, df in self.dfs.items()}

        idxs[ts_is_measured_key] = idxs["next_timepoint_was_measured"].copy()

        for idxs_k, dfs_k in (
            ("ts", "ts"),
            ("notes", "notes"),
            (ts_is_measured_key, "next_timepoint_was_measured"),
        ):
            if idxs_k in idxs:
                hours_in = self.dfs[dfs_k].index.get_level_values("hours_in")
                idxs[idxs_k] &= (hours_in >= start_time) & (hours_in < end_time)

        # get the next task for predictions
        for k in [
            "rolling_tasks_binary_multilabel",
            "rolling_tasks_multiclass",
            "rolling_ftseq",
            "next_timepoint",
            "next_timepoint_was_measured",
        ]:
            if k not in self.dfs or self.dfs[k] is None:
                continue
            if k in idxs:
                idxs[k] &= self.dfs[k].index.get_level_values("hours_in") == end_time

        # get the correct subset of the dfs
        dfs = {k: df.loc[idxs[k]].copy() for k, df in self.dfs.items() if df is not None}
        dfs[ts_is_measured_key] = self.dfs["next_timepoint_was_measured"].loc[idxs[ts_is_measured_key]].copy()

        # break up all of these dataframes that were processed as one into individual dfs
        for k in ("rolling_tasks_multiclass", "static_tasks_multiclass"):
            df = dfs[k]
            for c in df.columns:
                dfs[c] = df[[c]]

            del dfs[k]

        # For the next timepoint, we only want the means of measured labs.
        # TODO(mmd): Is this the right place for this logic? Or should it go earlier?
        cols = dfs["next_timepoint"].columns
        mean_labs_cols = [c for c in cols if type(c) is tuple and c[1] == "mean"]
        dfs["next_timepoint"] = dfs["next_timepoint"][mean_labs_cols].fillna(value=-1)

        dfs[ts_vals_key] = dfs["ts"].loc[:, mean_labs_cols].copy().fillna(0)

        dfs["ts"].loc[:, self.ts_continuous_cols] = self.impute_fn(dfs["ts"].loc[:, self.ts_continuous_cols]).fillna(
            0
        )  # First impute, then fill w/ 0.

        np_arrays = {k: df.values for k, df in dfs.items()}
        # np_arrays.pop('notes', None)

        # Now adding the mask key.
        if self.imputation_mask_rate > 0:
            any_masked = False
            while not any_masked:
                mask_prob = np.random.uniform(size=(self.max_seq_len, 1))
                any_masked = (mask_prob < self.imputation_mask_rate).sum() > 0
            np_arrays[imputation_mask_key] = np.where(
                mask_prob < self.imputation_mask_rate, np.ones_like(mask_prob), np.zeros_like(mask_prob)
            )

        # Padding
        for k in ("ts", ts_vals_key, ts_is_measured_key, "notes"):
            if k in np_arrays:
                num_features = np_arrays[k].shape[1]
                if np_arrays[k].shape[0] != self.max_seq_len:
                    if self.max_seq_len > seq_len:
                        pad = np.zeros((self.max_seq_len - seq_len, num_features))
                        np_arrays[k] = np.expand_dims(np.concatenate((np_arrays[k], pad)), 0)
                elif self.max_seq_len == seq_len:
                    np_arrays[k] = np.expand_dims(np_arrays[k], 0)

        np_arrays["tasks_binary_multilabel"] = np.concatenate(
            [np_arrays[k] for k in self.binary_multilabel_task_concat_order], axis=1
        )
        del np_arrays["rolling_tasks_binary_multilabel"]
        del np_arrays["static_tasks_binary_multilabel"]

        tensors = {}
        for k, arr in np_arrays.items():
            # Ensure all elements in the array are numeric
            arr = np.asarray(arr, dtype=np.float32 if np.issubdtype(arr.dtype, np.floating) else np.int64)

            if arr.shape[0] == 1:
                tensors[k] = torch.tensor(arr[0])
            else:
                tensors[k] = torch.tensor(arr)

        save_id_path = os.path.join(self.save_path, f"{icustay_id}.pt")
        torch.save({k: v.half() if v.dtype == torch.float32 else v for k, v in tensors.items()}, save_id_path)

        return tensors


def load_data():
    patient_data = pd.read_csv(patient_path)
    hospital_df = patient_data[["patientunitstayid", "hospitalid"]]  # 200859
    hospital_id = set(hospital_df["hospitalid"])

    unitstay_list = []

    hospital_stay_id_map = {}
    for i in hospital_id:
        patient_list = hospital_df[hospital_df["hospitalid"] == i]["patientunitstayid"].tolist()
        hospital_stay_id_map[i] = patient_list
        unitstay_list += patient_list

    unitstay_list = list(set(unitstay_list))

    return hospital_stay_id_map, unitstay_list


if __name__ == "__main__":

    directory_path = f"{args.data_path}/eicu-2.0/federated_preprocessed_data/final_datasets_for_sharing/dataset_eicu/rotations/no_notes"
    print("START CACHING...")
    for phase in ["train", "tuning", "test"]:
        all_data = pd.read_pickle(os.path.join(directory_path, "0", f"{phase}.pkl"))

        new_dataset = PatientDataset(**all_data)

        data_num = len(new_dataset)
        print(f"Phase: {phase} / Total number: {data_num}")

        for i in range(data_num):
            new_dataset[i]

    directory_path = f"{args.data_path}/eicu-2.0"
    hospital_path = os.path.join(directory_path, "hospital.csv")
    patient_path = os.path.join(directory_path, "patient.csv")
    preprocessed_dir = f"{args.data_path}/eicu-2.0/federated_preprocessed_data/final_datasets_for_sharing/dataset_eicu/rotations/no_notes"

    all_path = f"{args.data_path}/eicu-2.0/federated_preprocessed_data"

    hospital_info, unitstay_list = load_data()
    hospitals = list(hospital_info.keys())
    cache_dir = f"{args.data_path}/eicu-2.0/federated_preprocessed_data/cached_data"
    unitstay_id = os.listdir(cache_dir)

    for i, stay in enumerate(unitstay_id):
        unitstay_id[i] = int(stay.split(".")[0])

    preprocessed_hospital_data = {}

    for hospital in hospitals:
        hospital_stay_list = hospital_info[hospital]
        for stay in hospital_stay_list:
            if stay in unitstay_id:
                if hospital in preprocessed_hospital_data.keys():
                    preprocessed_hospital_data[hospital].append(stay)
                else:
                    preprocessed_hospital_data[hospital] = [stay]

    with open(os.path.join(all_path, "hospital_unitstay.json"), "w") as f:
        json.dump(preprocessed_hospital_data, f)

    post_hospital, unitstay = [], []

    for k in preprocessed_hospital_data.keys():
        post_hospital.append(k)
        unitstay.append(len(preprocessed_hospital_data[k]))

    new_dict = {"hospital": post_hospital, "unitstay": unitstay}
    new_frame = pd.DataFrame(new_dict)
    new_frame = new_frame.sort_values(by=["unitstay"], axis=0, ascending=False)
    new_frame.to_csv(os.path.join(all_path, "hospital_scale_info.csv"), index=False)

    print("Finish!")
