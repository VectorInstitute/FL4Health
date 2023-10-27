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
        
        # >>>>>>>>
        # TODO generate these data
        self.arithmetic_modulus = None
        self.client_integer = None  # 'integer' reminds us this is not cid from flwr
        self.reconstruction_threshold = None
        assert 1 <= self.client_integer <= self.arithmetic_modulus
        # <<<<<<<<

        self.crypto = ClientCryptoKit(client_integer=self.client_integer, 
                                      arithemetic_modulus=self.arithmetic_modulus,
                                      reconstruction_threshold=self.reconstruction_threshold)
        self.fl_round = 0  # federated round

    def fit(parmeters: NDArrays, config: Config) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        assert config['event_name'] == Event.MASKED_INPUT_COLLECTION.value

        # unmaked params
        update_params = super().fit(parmeters, config)

        # add masks


        return update_params

    def get_properties(self, config: Config) -> Dict[str, Scalar]:

        if not self.initialized:
            self.setup_client(config)

        # be sure to include a round in config
        self.fl_round = config["fl_round"]

        response_dict = {}

        match config['event_name']:
            case Event.ADVERTISE_KEYS.value:
                response_dict = self._generate_public_keys_dict()

            case Event.SHARE_KEYS.value:
                self.crypto.process_bobs_keys(bobs_keys_list=config['bobs_keys_list']) 

            case Event.MASKED_INPUT_COLLECTION:
                pass
            case Event.UNMASKING:
                pass
            case _ :
                response_dict = {"num_train_samples": self.num_train_samples, "num_val_samples": self.num_val_samples}

        return response_dict
    
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


    def api_receiver(self, config: Config) -> Any: 
        pass