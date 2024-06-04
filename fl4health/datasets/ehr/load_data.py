import os, wandb, json, torch
from torch.utils.data import DataLoader, RandomSampler, SubsetRandomSampler

import json, pickle, enum, copy, os, random, sys, torch, numpy as np, pandas as pd
from torch.utils.data import DataLoader, Dataset
from typing import Dict

############################################################################

class CachedDataset(Dataset): # Dataset
    def __init__(
        self,
        hospital_id,
        unitstays,
        data_path
    ):
        self.hospital_id = hospital_id
        self.unitstay = unitstays
        self.load_path = f"{data_path}/eicu-2.0/federated_preprocessed_data/cached_data"

    def __len__(self): 
        return len(self.unitstay)

    def __getitem__(self, item):
        """
        Returns:
            tensors (dict)
                'rolling_ftseq', None
                'ts', [batch_size, sequence_length, 165]
                'statics', [batch_size, 15]
                'next_timepoint',
                'next_timepoint_was_measured',
                'disch_24h', [batch_size, 10]
                'disch_48h', [batch_size, 10]
                'Final Acuity Outcome', [batch_size, 12]
                'ts_mask',
                'tasks_binary_multilabel', [batch_size, 3]
        """

        unitstay = self.unitstay[item]

        unitstay_path = os.path.join( self.load_path, f"{unitstay}.pt" )
        tensors = torch.load(unitstay_path)
        return tensors

def nan_checker( args, patients ):
    
    new_patients = []

    for patient in patients :
        temp = CachedDataset("total", [patient], args.data_path)

        if args.task == "mort_24h" or args.task == "mort_48h" or args.task == "LOS" :
            index = ["mort_24h", "mort_48h", "LOS"].index(args.task)
            label = temp[0]['tasks_binary_multilabel'][index]
        else :
            label = temp[0][args.task]

        check = torch.isnan(label).item()

        if check == False :
            new_patients.append(patient)

    return new_patients


def get_dataset(args):
    new_dir = f"{args.data_path}/eicu-2.0/federated_preprocessed_data/data_split_fixed"
    client_id = args.hospital_id
    
    train_dataset, valid_dataset, test_dataset = [], [], []

    for c_id in client_id :
        each_client = str(c_id)

        ##########################################################
        if args.task in ['mort_24h', 'mort_48h', 'LOS' ] :
            with open( os.path.join(new_dir, f"{c_id}.json"), "r") as json_file:
                json_data = json.load(json_file)
            train_patients = json_data['train']
            valid_patients = json_data['valid']
            test_patients = json_data['test']
        else :
            with open( os.path.join(new_dir, f"{c_id}_ver2.json"), "r") as json_file:
                json_data = json.load(json_file)
            train_patients = json_data['train']
            valid_patients = json_data['valid']
            test_patients = json_data['test']

        ##########################################################

        print(f"{c_id} client's patient number : ", len(train_patients + valid_patients + test_patients))

        train_patients = nan_checker( args, train_patients )
        valid_patients = nan_checker( args, valid_patients )
        test_patients = nan_checker( args, test_patients )

        train_dataset.append( CachedDataset( each_client, train_patients, args.data_path) )
        valid_dataset.append( CachedDataset( each_client, valid_patients, args.data_path) )
        test_dataset.append( CachedDataset( each_client, test_patients, args.data_path) )

    return client_id, train_dataset, valid_dataset, test_dataset


def get_dataloader(args, train_dataset, valid_dataset, test_dataset):

    train_loaders, valid_loaders, test_loaders = [], [], []
    client_weights = []

    for i in range(len(args.hospital_id)) :
        client_weights.append( len(train_dataset[i]) )

        train_sampler = RandomSampler(train_dataset[i])
        if len(train_dataset[i]) % args.batch_size == 1:
            train_loaders.append( DataLoader(train_dataset[i], sampler=train_sampler, batch_size=args.batch_size, num_workers=8, drop_last=True) )
        else :
            train_loaders.append( DataLoader(train_dataset[i], sampler=train_sampler, batch_size=args.batch_size, num_workers=8, drop_last=False) )

        valid_sampler = RandomSampler(valid_dataset[i])
        if len(valid_dataset[i]) % args.batch_size == 1 :
            valid_loaders.append( DataLoader(valid_dataset[i], sampler=valid_sampler, batch_size=args.batch_size, num_workers=8, drop_last=True) )
        else :
            valid_loaders.append( DataLoader(valid_dataset[i], sampler=valid_sampler, batch_size=args.batch_size, num_workers=8, drop_last=False) )

        if len(test_dataset[i]) % args.batch_size == 1 :
            test_loaders.append( DataLoader(test_dataset[i], batch_size=args.batch_size, num_workers=8, drop_last=True) )
        else :
            test_loaders.append( DataLoader(test_dataset[i], batch_size=args.batch_size, num_workers=8, drop_last=False) )


    total = sum(client_weights)
    client_weights = [ weight / total for weight in client_weights ]

    return client_weights, train_loaders, valid_loaders, test_loaders