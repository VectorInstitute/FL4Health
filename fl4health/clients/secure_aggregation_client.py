import pickle
from logging import DEBUG, INFO, WARN
from pathlib import Path
import random
import copy
from typing import Dict, Optional, Sequence, Tuple
import time
import torch
import numpy
from opacus.data_loader import DPDataLoader
import torch.utils
import gc
from torch.utils.data import Subset, DataLoader
from flwr.common.logger import log
from flwr.common.typing import Config, List, NDArrays, Scalar
import torch.utils.data

from fl4health.checkpointing.checkpointer import TorchModuleCheckpointer
from fl4health.clients.basic_client import BasicClient
from fl4health.parameter_exchange.parameter_exchanger_base import ParameterExchanger
from fl4health.parameter_exchange.secure_aggregation_exchanger import SecureAggregationExchanger
from fl4health.privacy_mechanisms.index import PrivacyMechanismIndex
from fl4health.security.secure_aggregation import ClientCryptoKit, ClientId, Event, ShamirSecret, ShamirSecrets
from fl4health.utils.losses import LossMeterType
from fl4health.utils.metrics import Metric
from fl4health.checkpointing.client_module import CheckpointMode, ClientCheckpointAndStateModule


from fl4health.privacy_mechanisms.slow_discrete_gaussian_mechanism import (
    discrete_gaussian_noise_vector,
    generate_random_sign_vector,
    generate_walsh_hadamard_matrix,
    pad_zeros,
    randomized_rounding,
    calculate_delta_squared,
    calculate_tau,
    single_fl_round_concentrated_dp,
    clip_vector,
    get_exponent
)
from fl4health.privacy_mechanisms.discrete_gaussian_mechanism import (
    generate_discrete_gaussian_vector,
    fwht,
    shift_transform
)
from fl4health.privacy_mechanisms.index import PrivacyMechanismIndex
from fl4health.servers.secure_aggregation_utils import get_model_norm, vectorize_model, get_model_layer_types, change_model_dtypes
import json 
import os
import uuid 
import timeit

from fl4health.privacy_mechanisms.gaussian_mechanism import gaussian_mechanism
from statistics import mean
from opacus import PrivacyEngine

from fl4health.reporting.base_reporter import BaseReporter



torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')
# torch.set_default_dtype(torch.float64)


class SecureAggregationClient(BasicClient):
    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
        device: torch.device,
        privacy_settings,
        loss_meter_type: LossMeterType = LossMeterType.AVERAGE,
        # metric_meter_type: MetricMeterType = MetricMeterType.AVERAGE,
        checkpointer: Optional[TorchModuleCheckpointer] = None,
        reporters: Sequence[BaseReporter] | None = None,
        progress_bar: bool = False,
        client_name: str | None = None,
        client_id: str = uuid.uuid1(),
        task_name: str = '',
        num_mini_clients = 8,
    ) -> None:
        
        super().__init__(data_path, metrics, device, loss_meter_type)
        
        

        self.client_id = client_id
        self.task_name = task_name

        self.num_mini_clients = num_mini_clients
        self.mini_clients_data_loaders: List[DataLoader]
        self.start_time:float

        temporary_dir = os.path.join(
            os.path.dirname(checkpointer.checkpoint_path),
            'temp'
        )
        self.temporary_dir = temporary_dir

        if not os.path.exists(temporary_dir):
            os.makedirs(temporary_dir)

        # path for model vector 
        self.temporary_model_path = os.path.join(
            temporary_dir,
            f'client_{self.client_id}_initial_model.pth'
        )

        # path to torch.save() model
        # NOTE saving state dict can help avoid Opacus errors.
        self.temporary_model_state_path = os.path.join(
            temporary_dir,
            f'client_{self.client_id}_initial_model_state.pth'
        )
        self.temporary_optimizer_state_path = os.path.join(
            temporary_dir,
            f'client_{self.client_id}_optimizer_state.pth'
        )

        metrics_dir = os.path.join(
            os.path.dirname(checkpointer.checkpoint_path),
            'metrics'
        )

        if not os.path.exists(metrics_dir):
            os.makedirs(metrics_dir)

        self.metrics_path = os.path.join(
            metrics_dir,
            f'client_{self.client_id}_metrics.json'
        )

        # client-side cryptography for Secure Aggregation
        self.crypto = ClientCryptoKit()
        self.dropout_mode = None

        self.parameter_exchanger = SecureAggregationExchanger()
        log(INFO, f"Client initializes parameter exchange as {type(self.parameter_exchanger)}")

        # set after model initialization
        self.model_dim = None
        self.padded_model_dim = None

        # TODO set differential privacy parameters
        self.privacy_settings = {
            **privacy_settings,
            "dp_mechanism": PrivacyMechanismIndex.DiscreteGaussian.value,
        }

        with open(self.metrics_path, 'w+') as file:
            json.dump({
                'task_name': self.task_name,
                'client_id': self.client_id,
            },file)

        assert 0 <= self.privacy_settings["bias"] < 1

        self.debug_mode = True


    # The 'main' function for client-side secure aggregation.
    def get_properties(self, config: Config) -> Dict[str, Scalar]:
        """Receiver of server calls for Secure Aggregation Protocol."""

        if not self.initialized:
            self.setup_client(config)

        match config["event_name"]:

            case Event.ADVERTISE_KEYS.value:

                # NOTE this client integer ID currently persists across SecAgg rounds
                self.crypto.set_client_integer(integer=config["client_integer"])


                # these determine the number of Shamir shares
                self.crypto.set_number_of_bobs(integer=config["number_of_bobs"])
                self.crypto.set_reconstruction_threshold(new_threshold=config["shamir_reconstruction_threshold"])

                # modulus may change at the start of each SecAgg if dropout occurs (refer to documentation)
                self.crypto.set_arithmetic_modulus(modulus=config["arithmetic_modulus"])
                self.privacy_settings['arithmetic_modulus'] = self.crypto.arithmetic_modulus

                with open(self.metrics_path, 'r') as file:
                    metrics_to_save = json.load(file)
                    metrics_to_save['privacy_settings'] = self.privacy_settings

                with open(self.metrics_path, 'w') as file:
                    json.dump(metrics_to_save, file)

                self.dropout_mode = config["dropout_mode"]

                public_keys = self.crypto.generate_public_keys()

                response_dict = {
                    # main data
                    "client_integer": self.crypto.client_integer,
                    "public_encryption_key": public_keys.encryption_key,
                    "public_mask_key": public_keys.mask_key,
                    # for server-side validation
                    "event_name": Event.ADVERTISE_KEYS.value,
                }

            case Event.SHARE_KEYS.value:

                unload_keys = pickle.loads(config["bobs_public_keys"])
                unload_keys.pop(self.crypto.client_integer)  # remove Alice herself

                t = self.crypto.reconstruction_threshold
                if len(unload_keys) < t:

                    error_msg = f"""
                    Too many droped out clients (#peers){len(unload_keys)} < threshold {t},
                    aborting client {self.crypto.client_integer}.
                    """

                    log(WARN, error_msg)

                    # NOTE open problem | when one client triggers everyone to abort FL,
                    # how does server know it's not malicious?
                    exit()

                # key agreement and storage
                self.crypto.register_bobs_keys(bobs_keys_dict=unload_keys)

                # ****************** debug point (starts) *******************

                # log(DEBUG, self.crypto.agreed_mask_keys)
                # log(DEBUG, self.crypto.agreed_encryption_keys)

                # ****************** debug point (ends) *********************

                # generate self-mask seed
                self.crypto.set_self_mask_seed()

                # Shamir shares for 1) self-mask and 2) pairmask secrete key
                shamir_pair_self = self.crypto.get_encrypted_shamir_shares()

                response_dict = {
                    "event_name": Event.SHARE_KEYS.value,
                    "client_integer": self.crypto.client_integer,
                    "serialized_encrypted_shamir": pickle.dumps(shamir_pair_self),
                }

            case Event.MASKED_INPUT_COLLECTION.value:

                self.crypto.bob_shamir_secrets = {}
                received = pickle.loads(config["pickled_message"])  # expects dict

                t = self.crypto.reconstruction_threshold
                if len(received) < t:

                    error_msg = f"""
                    Too many droped out clients (#peers){len(received)} < threshold {t},
                    aborting client {self.crypto.client_integer}.
                    """

                    log(WARN, error_msg)

                    # NOTE triggers everyone to abort FL,
                    exit()

                # receive Shamir shares from other clients
                self.crypto.register_shamir_shares(shamir_shares=received)

                response_dict = {
                    "event_name": Event.MASKED_INPUT_COLLECTION.value,
                    "client_integer": self.crypto.client_integer,
                }

            case Event.UNMASKING.value:
                # this case is not executed if we assume no dropouts (hence no need for mask removal)

                # NOTE we do not check U4 is a subset of U3
                # since we opt out of Round 3 (Consistency Check) of SecAgg
                self.debugger("oh my good santa")
                pairmask_shamir_secrets: Dict[ClientId, ShamirSecret] = {}
                for id in pickle.loads(config["pickled_dropout_clients"]):
                    # NOTE these secrets should be unencrypted
                    pairmask_shamir_secrets[id] = self.crypto.bob_shamir_secrets[id].pairwise

                selfmask_shamir_secrets: Dict[ClientId, ShamirSecret] = {}
                for id in pickle.loads(config["pickled_online_clients"]):
                    # NOTE these secrets should be unencrypted
                    selfmask_shamir_secrets[id] = self.crypto.bob_shamir_secrets[id].individual

                response_dict = {
                    "event_name": Event.UNMASKING.value,
                    "current_fl_round": config["current_fl_round"],
                    "pickled_pairmask_shamir_secrets": pickle.dumps(pairmask_shamir_secrets),
                    "pickled_selfmask_shamir_secrets": pickle.dumps(selfmask_shamir_secrets),
                }
                self.debugger("oh my good santa end")
            case _:
                response_dict = {
                    "num_train_samples": self.num_train_samples,
                    "num_val_samples": self.num_val_samples,
                    "response": "client served default",
                }

        return response_dict
    
    def setup_opacus(self) -> None:
        privacy_engine = PrivacyEngine()

        # hard coded in for now
        self.noise_multiplier = 1e-16
        self.clipping_bound = 1e16

        # Apply PrivacyEngine to model, optimizer, and train_loader
        # self.model, self.optimizer, self.train_loader = privacy_engine.make_private(
        #     module=self.model,
        #     optimizer=self.optimizer,
        #     data_loader=self.train_loader,
        #     noise_multiplier=self.noise_multiplier,
        #     max_grad_norm=self.clipping_bound,
        #     clipping="flat",
        #     poisson_sampling=False,
        # )
        log(DEBUG, f'train_loader length in `setup_opacus`: {len(self.train_loader.dataset)}')

        # Get the total size of the dataset
        total_size = len(self.train_loader.dataset)
        client_size = total_size // self.num_mini_clients   # Determine size per client

        # Split the dataset indices for mini-clients
        indices = list(range(total_size))
        mini_clients_data_loaders = []

        # Create DPDataLoaders for each mini-client
        for i in range(self.num_mini_clients):
            start_idx = i * client_size
            end_idx = (i + 1) * client_size if i != self.num_mini_clients - 1 else total_size
            client_indices = indices[start_idx:end_idx]

            # Create a Subset for each client and corresponding DataLoader
            client_subset = Subset(self.train_loader.dataset, client_indices)

            # Here we use DPDataLoader without batch_size and shuffle, as these conflict with batch_sampler
            # client_data_loader = DPDataLoader(client_subset, sample_rate= 40 / client_size)
            generator = torch.Generator(device='cuda')  # Set the generator to use 'cuda'

            # Create the DataLoader with the specified generator
            client_data_loader = DataLoader(
                client_subset,
                batch_size=self.train_loader.batch_size,
                shuffle=True,
                num_workers=self.train_loader.num_workers,
                generator=generator  # Pass the generator here
            )
            mini_clients_data_loaders.append(client_data_loader)
        
        # Store the list of DataLoaders for clients
        self.mini_clients_data_loaders = mini_clients_data_loaders
    

    # Orchestrates training
    def fit(self, parameters: NDArrays, config: Config) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        local_epochs, local_steps, current_server_round, _, _ = self.process_config(config)
        self.current_server_round = current_server_round
        log(INFO, f' start of client server round {current_server_round}')
        if not self.initialized:
            self.setup_client(config)
            self.optimizer = self.get_optimizer(config)
            self.setup_opacus()

        # local model <- global model
        self.set_parameters(parameters, config, True)

        # store initial model vector for computing model delta
        vector_0 = vectorize_model(self.model)
        torch.save(vector_0, self.temporary_model_path)
        del vector_0

        # TODO round 0 is a temporary work around for server to sample clients
        # NOTE when removing this round, be sure to initialize parameters such as model_dim elsewhere
        if current_server_round == 0:
            
            self.start_time=timeit.default_timer()

            # set dimension
            self.model_dim = sum(param.numel() for param in self.model.state_dict().values())
            self.padded_model_dim = 2**get_exponent(self.model_dim)
            
            # freeze model layer dtypes 
            # (these dtypes are modified during SecAgg and need to be changed back to avoid errors!)
            self.layer_dtypes = get_model_layer_types(self.model)

            # NOTE this is a full exchanger, needs to be modified for any partial exchanger
            parmeters = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
            metrics = {}

            return (
                parmeters,
                self.num_train_samples,
                metrics,
            )

        
        # for param in self.model.parameters():
        #     param.data = param.data.to(torch.float32)

        # SecAgg changes dtypes to higher precision float
        # this change needs to be reverted before training to avoid error
        try:
            self.revert_layer_dtype()
        except AttributeError:
            print("AttributeError encontered. Not reverting layer dtype")

        # log(INFO, f'-----model dtype list round {current_server_round} after reverting--------')
        # log(INFO, get_model_layer_types(self.model)==self.layer_dtypes)

        # for name, layer in self.model.state_dict().items():
        #     log(DEBUG, f'current_round: {current_server_round}')
        #     log(DEBUG, name)
        #     log(DEBUG, layer.dtype)

        # miniclients metrics and losses
        collective_metrics = {}
        collective_losses = {}
        cumulative_delta = None  # To hold cumulative delta of mini-clients
        initial_state_dict = copy.deepcopy(self.model.state_dict())
        torch.save(self.model.state_dict(), self.temporary_model_state_path)
        torch.save(self.optimizer.state_dict(), self.temporary_optimizer_state_path)

        if local_epochs is not None:
            # initial model and optimizer
            torch.save(self.model.state_dict(), self.temporary_model_state_path)
            torch.save(self.optimizer.state_dict(), self.temporary_optimizer_state_path)

            for id in range(1, 1+ self.num_mini_clients):
                log(INFO, f'Training by epochs: {local_epochs} local epochs')
                self.model.load_state_dict(torch.load(self.temporary_model_state_path))
                # self.optimizer = self.get_optimizer(config)
                self.optimizer.load_state_dict(torch.load(self.temporary_optimizer_state_path))
                
                self.train_loader = self.mini_clients_data_loaders[id - 1]
                log(DEBUG, f'train_loader length: {len(self.train_loader.dataset)}')

                loss_dict, metrics, training_set_size = self.train_by_epochs(local_epochs, current_server_round)

                for k, v in metrics.items():
                    if k not in collective_metrics:
                        collective_metrics[k] = [v]
                    else:
                        collective_metrics[k].append(v)

                for k, v in loss_dict.items():
                    if k not in collective_losses:
                        collective_losses[k] = [v]
                    else:
                        collective_losses[k].append(v)

                # for name, param in current_state_dict.items():
                #     initial_param = initial_state_dict[name]
                #     param_change = torch.norm(param - initial_param).item()
                #     log(INFO, f'Parameter change for {name} after mini-client {id}: {param_change}')
                # del current_state_dict

                # for name, param in self.model.named_parameters():
                #     if param.grad is not None:
                #         log(INFO, f'Mini-client {id}, {local_steps} local steps with gradient mean: {param.grad.mean().item()}')

                trained_mini_model_vector = vectorize_model(self.model)
                path = os.path.join(self.temporary_dir,f'client_{self.client_id}_mini_client_{id}.pth')
                torch.save(trained_mini_model_vector, path)
                del trained_mini_model_vector

            # assumes no sampling: train size is constant for each miniclient
            self.num_train_samples = self.num_mini_clients * training_set_size

        # elif local_steps is not None:
        #     # initial model
        #     torch.save(self.model.state_dict(), self.temporary_model_state_path)
        #     torch.save(self.optimizer.state_dict(), self.temporary_optimizer_state_path)
        #     for id in range(1, 1+ self.num_mini_clients):
        #         log(INFO, f'Mini-client {id} training by steps: {local_steps} local steps')
        #         self.model.load_state_dict(torch.load(self.temporary_model_state_path))
        #         self.optimizer = self.get_optimizer(config)
        #         self.optimizer.load_state_dict(torch.load(self.temporary_optimizer_state_path))

        #         # del self.train_loader  # Delete reference to the previous loader
        #         # torch.cuda.empty_cache()  # Clear any memory allocated by CUDA
        #         # gc.collect()  # Trigger Python garbage collection
                
        #         self.train_loader = self.mini_clients_data_loaders[id - 1]
        #         log(DEBUG, f'train_loader length: {len(self.train_loader.dataset)}')

        #         # appropriately aggregate these metrics and losses
        #         loss_dict, metrics, training_set_size = self.train_by_steps(local_steps, current_server_round)

        #         for k, v in metrics.items():
        #             if k not in collective_metrics:
        #                 collective_metrics[k] = [v]
        #             else:
        #                 collective_metrics[k].append(v)

        #         for k, v in loss_dict.items():
        #             if k not in collective_losses:
        #                 collective_losses[k] = [v]
        #             else:
        #                 collective_losses[k].append(v)

        #         # for name, param in self.model.named_parameters():
        #         #     if param.grad is not None:
        #         #         log(INFO, f'Mini-client {id}, {local_steps} local steps with gradient mean: {param.grad.mean().item()}')

        #         trained_mini_model_vector = vectorize_model(self.model)
        #         path = os.path.join(self.temporary_dir,f'client_{self.client_id}_mini_client_{id}.pth')
        #         torch.save(trained_mini_model_vector, path)
        #         del trained_mini_model_vector

        #     # assumes no sampling: train size is constant for each miniclient
        #     self.num_train_samples = self.num_mini_clients * training_set_size
        else:
            raise ValueError("Must specify either local_epochs or local_steps in the Config.")

        mean_metrics = {}
        for k, v in collective_metrics.items():
            mean_metrics[k] = mean(v)

        mean_losses = {}
        for k, v in collective_losses.items():
            mean_losses[k] = mean(v)
            


        # Update after train round (Used by Scaffold and DP-Scaffold Client to update control variates)
        self.update_after_train(local_steps, mean_losses)

        # NOTE uncomment for debugging
        # pickle.dump(self.model.state_dict(), open(f"examples/secure_aggregation_example/local_models/{random()}.pkl", "wb"))
        
        # log(INFO, f'Number of training examples: {self.num_train_samples}')

        if self.privacy_settings["dp_mechanism"] == PrivacyMechanismIndex.DiscreteGaussian.value:
            m = self.crypto.arithmetic_modulus
            assert isinstance(m, int) and m > 1
            self.privacy_settings["arithmetic_modulus"] = m
        else:
            dp_kind = self.privacy_settings["dp_mechanism"]
            log(WARN, f"The DP-mechanism you chose {dp_kind} is not yet implemented. Aborting FL.")
            exit()
        
        metrics_to_save = {}

        with open(self.metrics_path, 'r') as file:
            metrics_to_save = json.load(file)
            metrics_to_save['model_size'] = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            metrics_to_save['current_round'] = current_server_round

            for key, value in mean_metrics.items():
                if key not in metrics_to_save:
                    metrics_to_save[key] = [value]
                else:
                    metrics_to_save[key].append(value)

            for key, value in mean_losses.items():
                if key not in metrics_to_save:
                    metrics_to_save[key] = [value]
                else:
                    metrics_to_save[key].append(value)
            
            # if 'arithmetic_modulus' not in metrics_to_save:
            #     metrics_to_save['arithmetic_modulus'] = [self.privacy_settings['arithmetic_modulus']]
            # else:
            #     metrics_to_save['arithmetic_modulus'].append(self.privacy_settings['arithmetic_modulus'])


            if 'time' not in metrics_to_save:
                metrics_to_save['time'] = [timeit.default_timer()-self.start_time]
            else:
                metrics_to_save['time'].append(timeit.default_timer()-self.start_time)
                
            # metrics_to_save[current_server_round] = {
            #     'round': current_server_round,
            #     'metrics': metrics,
            #     'arithmetic_modulus': self.privacy_settings['arithmetic_modulus'],
            # }

        with open(self.metrics_path, 'w') as file:
            json.dump(metrics_to_save, file)
            # log(DEBUG, f'finished recording metrics for round {current_server_round}')

        return (
            self.secure_and_privatize(),
            self.num_train_samples,
            mean_metrics,
        )
    
    def secure_and_privatize(self) -> NDArrays:
        vector = self.process_model_post_training(mini_client_id=1)
        for id in range(2, 1+self.num_mini_clients):
            vector += self.process_model_post_training(mini_client_id=id)
            # DEBUG
            log(INFO, f'Arithmetic modulus = {self.crypto.arithmetic_modulus}')
            vector %= self.crypto.arithmetic_modulus

        mask = torch.tensor(self.crypto.get_pair_mask_sum(vector_dim=self.padded_model_dim, allow_dropout=False))
        # self.echo('mask', mask)
        # vector *= 0
        vector += mask
        vector %= self.crypto.arithmetic_modulus

        vector_np = vector.cpu().numpy()

        # NOTE find average delta or max delta
        delta = vector_np

        return [vector_np, delta, vectorize_model(self.model).cpu().numpy()]
    
    def process_model_post_training(self, mini_client_id: int) -> torch.Tensor:
        """ We send model delta to the server. See Algorithm 1 in 
        Ref
            The Distributed Discrete Gaussian Mechanism for Federated Learning with Secure Aggregation
            https://arxiv.org/pdf/2102.06387.pdf
        
        """

        assert self.model is not None and self.parameter_exchanger is not None
        self.parameter_exchanger = SecureAggregationExchanger()
        # torch.set_default_dtype(torch.float64)

        # reshape model tensors and concatenate into a vector
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        vector_0: torch.Tensor = torch.load(self.temporary_model_path).to(device=device)

        mini_client_model_path = os.path.join(self.temporary_dir,f'client_{self.client_id}_mini_client_{mini_client_id}.pth')
        vector_1: torch.Tensor = torch.load(mini_client_model_path).to(device=device)
        # self.echo('trained model', vector_1)
        # log(INFO, f'vector_0 dtyle: {vector_0.dtype}, numel {vector_0.numel()}')
        # log(INFO, f'vector_1 dtyle: {vector_1.dtype} numel {vector_1.numel()}')

        # model diff
        vector = vector_1 - vector_0

        # vector = vectorize_model(self.model).to(device=device)
        delta = pad_zeros(vector=vector, dim=self.model_dim)
        del vector_0, vector_1
        
        # debug 
        # vector = torch.ones(self.model_dim)

        vector = vector.to(dtype=torch.float64)

        # log(INFO, f'post processing vector dtyle: {vector.dtype}')
        # log(DEBUG, 'model diff---------')
        # log(DEBUG, vector[:100])

        # for param in self.model.parameters():
        #     log(INFO, f'post processing model dtyle: {param.data.dtype}')
        #     break
        # log(DEBUG, 'clipped')
        # log(DEBUG, vector[:100])
        # TODO adjust clip to weighting
        # vector *= weight # weight for FedAvg, server divides out sum of client weights

        c, g = self.privacy_settings['clipping_threshold'], self.privacy_settings['granularity']
        
        # adjust for scaling of model vector by the weight (often the train data size)
        # weighted_clip = c * weight

        # NOTE the vector should already be clipped by opacus 
        vector = clip_vector(vector=vector, clip=c, granularity=g)
        mini_client_size = len(self.train_loader.dataset)
        vector = vector / self.num_mini_clients

        # log(DEBUG, 'weighted')
        # log(DEBUG, vector[:100])
        # x'' = H(Dx')
        # log(INFO, f'original vector {len(vector)}')
        vector = pad_zeros(vector=vector, dim=self.model_dim) # pad zeros 
        sign_vector = generate_random_sign_vector(dim=self.padded_model_dim)
        # log(INFO, f'padded vector {len(vector)}, signed vector {len(sign_vector)}, self.model_dim {self.model_dim}, self.padded_model_dim {self.padded_model_dim}')
        vector = torch.mul(vector, sign_vector) # hadamard product
        # log(DEBUG,'after zero pad')
        # log(INFO, vector)
        log(DEBUG, f'Starting Welsh Hadamard Transform')
        t0 = time.perf_counter()
        vector = fwht(vector)
        # log(DEBUG,'after WHT')
        # log(INFO, vector)
        t1 = time.perf_counter()
        log(DEBUG, f'Welsh Hadamard Transform finished in {t1-t0} sec')
        # vector = torch.matmul(
        #     input=generate_walsh_hadamard_matrix(exponent=get_exponent(self.model_dim)),
        #     other=vector
        # )
        # log(DEBUG, f'Starting randomized rounding')

        # b = self.privacy_settings['bias']
        # vector = randomized_rounding(vector=vector, clip=c, granularity=g, unpadded_model_dim=self.model_dim, bias=b)
        # log(DEBUG, f'Done rounding')


        # discrete Gaussian noise 
        # vector = torch.round(vector)
        b = self.privacy_settings['bias']
        # self.echo('before rounding, after fwht', vector)
        delta_squared = calculate_delta_squared(c, g, self.model_dim, b, mini_client_size)
        vector = randomized_rounding(vector, delta_squared, g)
        # self.echo('raw model server', vector)
        # log(DEBUG, f'casted type: {vector.dtype}')
        v = (self.privacy_settings['noise_scale'] / g) ** 2
        # log(INFO, f'Adding noise')

        # noise = torch.from_numpy(generate_discrete_gaussian_vector(dim=self.padded_model_dim, variance=v)).to(device='cuda' if torch.cuda.is_available() else 'cpu')
        # self.echo('noise l_infinity_norm', torch.linalg.vector_norm(noise.to(torch.float64), ord=float('inf')))
        # vector += noise

        # log(INFO, f'Adding mask')
        # log(DEBUG,'after noising')
        # log(INFO, vector)
        # TODO if dropout is turned on, then add selfmask below
        if mini_client_id == self.num_mini_clients:
            tau = calculate_tau(g, self.privacy_settings['noise_scale'], mini_client_size * 3)
            single_fl_round_concentrated_dp(delta_squared, self.model_dim, self.privacy_settings['noise_scale'], tau, mini_client_size * 3)
            
        return vector

    def generate_mask(self):
        dim = sum(param.numel() for param in self.model.parameters())
        # computes masking vector; this can only be run after masking seed agreement
        # modify to add self masking
        self.debugger(f"client integer {self.crypto.client_integer}", self.crypto.agreed_mask_keys)

        if self.dropout_mode:
            return self.crypto.get_duo_mask()

        pair_mask_vect: List[int] = self.crypto.get_pair_mask_sum(vector_dim=dim)
        return pair_mask_vect

    def constant_parameters(self, n=0) -> None:
        """Sets all parms to constant. For testing and pre-training init (fl_round=0) only."""
        zero_param_dict = self.model.state_dict()
        for layer_name, params in zero_param_dict.items():
            zero_param_dict[layer_name] = n * torch.ones(params.shape).to(torch.int64)
            # self.debugger(f'round {self.fl_round} >>>>', zero_param_dict[layer_name].dtype, zero_param_dict[layer_name].size())
        self.model.load_state_dict(zero_param_dict)

    def modify_parmeters(self):
        "Use this method to post process model parameters (i.e. masking & noising) after train and distributed evaluation."

        # counts trainable model parameters
        dim = sum(param.numel() for param in self.model.parameters())

        # computes masking vector; this can only be run after masking seed agreement
        pair_mask_vect: List[int] = self.crypto.get_pair_mask_sum(
            vector_dim=dim
        )  # pass in online_clients kwarg for drop out case
        # self.debugger('masking vector', pair_mask_vect)
        # modify parms  << quantization for privacy mechanism + modular arithmetic>>
        # pair_mask_vect = SkellamMechanism(query_vector=pair_mask_vect, skellam_variance=10)
        # pair_mask_vect = list(range(dim)) # TODO testing only
        i = 0
        params_dict = self.model.state_dict()
        for name, params in params_dict.items():
            j = i + params.numel()
            mask = torch.tensor(pair_mask_vect[i:j], dtype=torch.float64).reshape(params.size())
            params_dict[name] = 0 * torch.ones(
                params.shape, dtype=torch.float64
            )  # TEST ONLY: remove zeroing of params
            params_dict[name] += mask
            i = j

        # load modified parms
        self.model.to(torch.float64)
        self.model.load_state_dict(params_dict)
        # self.debugger(self.model.state_dict().values())

    def get_parameter_exchanger(self, config: Config) -> ParameterExchanger:
        """
        Returns Full Parameter Exchangers. Subclasses that require custom Parameter Exchangers can override this.
        """
        return SecureAggregationExchanger()
    

    def debugger(self, *info):
        log(DEBUG, 6 * "\n")
        for item in info:
            log(DEBUG, item)

    def _generate_public_keys_dict(self):
        keys = self.crypto.generate_public_keys()

        package = {
            # meta data
            "event_name": Event.ADVERTISE_KEYS.value,
            "fl_round": self.fl_round,
            "client_integer": self.client_integer,
            # important data
            "encryption_key": keys.encryption_key,
            "mask_key": keys.mask_key,
        }

        return package

    def get_metadata(self, event_name: Optional[str] = None) -> Dict[str, Scalar]:
        metadata = {
            "sender": "client",
            "client_integer": self.crypto.client_integer,
            "sec_agg_round": self.fl,
        }
        if event_name:
            metadata["event_name"] = event_name
        return metadata

    def train_by_epochs(
        self, epochs: int, current_round: Optional[int] = None
    ) -> Tuple[Dict[str, float], Dict[str, Scalar], int]:
        """These are cutomized for Poisson subsampling"""
        self.model.train()
        local_step = 0

        datasize = 0
        for local_epoch in range(epochs):
            log(INFO, f'Consumed {datasize} datapoints by epoch {local_epoch}.')
            self.train_metric_manager.clear()
            self.train_loss_meter.clear()
            for input, target in self.train_loader:

                datasize += list(input.shape)[0]

                input, target = input.to(self.device), target.to(self.device)
                losses, preds = self.train_step(input, target)
                self.train_loss_meter.update(losses)
                # self.train_metric_meter_mngr.update(preds, target)
                self.update_metric_manager(preds, target, self.train_metric_manager)
                
                self.update_after_step(local_step)
                self.total_steps += 1
                local_step += 1
            metrics = self.train_metric_manager.compute()
            losses = self.train_loss_meter.compute()
            loss_dict = losses.as_dict()

            # Log results and maybe report via WANDB
            self._log_results(loss_dict, metrics, current_round=current_round, current_epoch=local_epoch)
            self._log_results(loss_dict, metrics, current_round=current_round)

        # Return final training metrics
        return loss_dict, metrics, datasize

    def train_by_steps(
        self, steps: int, current_round: Optional[int] = None
    ) -> Tuple[Dict[str, float], Dict[str, Scalar], int]:
        """These are cutomized for Poisson subsampling"""

        # log(INFO, '===== training by steps ======')
        # for k, v in self.model.state_dict().items():
        #     log(INFO, v)

        self.model.train()

        # Pass loader to iterator so we can step through train loader
        train_iterator = iter(self.train_loader)

        self.train_loss_meter.clear()
        self.train_metric_manager.clear()

        datasize = 0
        for step in range(steps):
            # log(INFO, f'Consumed {datasize} datapoints by step {step}.')
            try:
                input, target = next(train_iterator)
            except StopIteration:
                # StopIteration is thrown if dataset ends
                # reinitialize data loader
                train_iterator = iter(self.train_loader)
                input, target = next(train_iterator)

            datasize += list(input.shape)[0]
            input, target = input.to(self.device), target.to(self.device)
            losses, preds = self.train_step(input, target)
            self.train_loss_meter.update(losses)
            # self.train_metric_meter_mngr.update(preds, target)
            self.update_metric_manager(preds, target, self.train_metric_manager)
            self.update_after_step(step)
            self.total_steps += 1

        losses = self.train_loss_meter.compute()
        loss_dict = losses.as_dict()
        metrics = self.train_metric_manager.compute()

        # Log results and maybe report via WANDB
        self._log_results(loss_dict, metrics, current_round=current_round)
        self._log_results(loss_dict, metrics, current_round=current_round)

        return loss_dict, metrics, datasize
    
    def revert_layer_dtype(self):
        self.model = change_model_dtypes(model=self.model, dtypes_list=self.layer_dtypes)

    def echo(self, message: str, to_print: any) -> None:
        if self.debug_mode:
            log(INFO, f'round {self.current_server_round}')
            log(INFO, f'{message}')
            log(INFO, to_print)
