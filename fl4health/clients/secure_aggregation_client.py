from fl4health.clients.basic_client import BasicClient
from pathlib import Path
from typing import Sequence, Optional, Tuple, Dict, Any 
from fl4health.utils.metrics import Metric, MetricMeter, MetricMeterManager, MetricMeterType
import torch
import torch.nn as nn
from fl4health.utils.losses import Losses, LossMeter, LossMeterType
from fl4health.checkpointing.checkpointer import TorchCheckpointer
from flwr.common.typing import Config, NDArrays, Scalar, List
from logging import DEBUG, INFO, WARNING
from flwr.common.logger import log
from fl4health.security.secure_aggregation import ClientCryptoKit, Event



class SecureAggregationClient(BasicClient):
    
    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
        device: torch.device,
        loss_meter_type: LossMeterType = LossMeterType.AVERAGE,
        metric_meter_type: MetricMeterType = MetricMeterType.AVERAGE,
        checkpointer: Optional[TorchCheckpointer] = None,
    ) -> None:
        
        super().__init__(data_path, metrics, device, loss_meter_type, metric_meter_type, checkpointer)
        
        # handles SecAgg cryptography on the client-side
        self.crypto = ClientCryptoKit()
        
        # federated round
        self.fl_round = 0  

    # def fit(parmeters: NDArrays, config: Config) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
    #     assert config['event_name'] == Event.MASKED_INPUT_COLLECTION.value

    #     # unmaked params
    #     update_params = super().fit(parmeters, config)

    #     # add masks


    #     return update_params

    def get_properties(self, config: Config) -> Dict[str, Scalar]:

        if not self.initialized:
            self.setup_client(config)
        # be sure to include a round in config
        # self.fl_round = config["fl_round"]

        response_dict = {}

        match config['event_name']:

            case Event.ADVERTISE_KEYS.value:
                assert config['sender'] == 'server'
                self.fl_round = config['fl_round']
                self.crypto.set_client_integer(integer=config['client_integer'])
                self.crypto.set_number_of_bobs(integer=config['number_of_bobs'])
                self.crypto.set_reconstruction_threshold(new_threshold= config['shamir_reconstruction_threshold'])
                self.crypto.set_arithmetic_modulus(modulus=config['arithmetic_modulus'])

                public_keys = self.crypto.generate_public_keys()

                response_dict = {
                    # meta data used for safe checking
                    'sender' : f'client',
                    'fl_round' : self.fl_round,
                    'client_interger' : self.crypto.client_integer,
                    'event_name' : Event.ADVERTISE_KEYS.value,

                    # main data
                    'public_encryption_key' : public_keys.encryption_key,
                    'public_mask_key' : public_keys.mask_key,
                }

            case Event.SHARE_KEYS.value:
                all_public_keys = config['bobs_public_keys']
                self.crypto.register_bobs_keys(bobs_keys_list=all_public_keys)

            case Event.MASKED_INPUT_COLLECTION.value:
                pass

            case Event.UNMASKING.value:
                pass
            case _ :
                response_dict = {"num_train_samples": self.num_train_samples, "num_val_samples": self.num_val_samples, "response": "client served default"}

        return response_dict
    
    def debugger(self, *info):
        log(DEBUG, 6*'\n')
        for item in info:
            log(DEBUG, item)
    
    def _generate_public_keys_dict(self):

        keys = self.crypto.generate_public_keys()

        package = {
            # meta data
            'event_name': Event.ADVERTISE_KEYS.value,   
            'fl_round': self.fl_round,
            'client_integer': self.client_integer,

            # important data
            'encryption_key': keys.encryption_key,
            'mask_key': keys.mask_key
        }

        return package