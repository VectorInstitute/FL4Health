from flwr.common.parameter import ndarrays_to_parameters
from flwr.common.typing import Config, Parameters
from torch.nn import Module
import math

from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import NaturalIdPartitioner
from fl4health.utils.dataset import TensorDataset


# the following function is consumed by the server strategy
def generate_config(local_epochs: int, batch_size: int, current_server_round: int) -> Config:
    package = {
        "local_epochs": local_epochs,
        "batch_size": batch_size,
        "current_server_round": current_server_round,
    }

    return package


# the following function is consumed by the server strategy
def get_parameters(model: Module) -> Parameters:
    return ndarrays_to_parameters([val.cpu().numpy() for _, val in model.state_dict().items()])

# def get_train_and_val_mnist_datasets(
#     data_dir: Path,
#     transform: Callable | None = None,
#     target_transform: Callable | None = None,
#     validation_proportion: float = 0.2,
#     hash_key: int | None = None,
# ) -> tuple[TensorDataset, TensorDataset]:
#     data, targets = get_mnist_data_and_target_tensors(data_dir, True)

#     train_data, train_targets, val_data, val_targets = split_data_and_targets(
#         data, targets, validation_proportion, hash_key
#     )

#     training_set = TensorDataset(train_data, train_targets, transform=transform, target_transform=target_transform)
#     validation_set = TensorDataset(val_data, val_targets, transform=transform, target_transform=target_transform)
#     return training_set, validation_set

if __name__ == '__main__':
    fds = FederatedDataset(
        dataset="flwrlabs/femnist",
        partitioners={"train": NaturalIdPartitioner(partition_by="writer_id")}
    )
    partition = fds.load_partition(partition_id=0)
    # print(partition)
    # print(fds.partitioners['train'].num_partitions)
    # partition2 = fds.load_partition(partition_id=0, split="train")
    # print(partition2)
    # partition3 = fds.load_partition(partition_id=0, split="train")
    # print(partition3)

    split_dict = partition.train_test_split(test_size = 0.2)
    train, test = split_dict['train'], split_dict['test']
    # print(train, test)
    # print(partition.features)

    transforms = ToTensor()

    # train = train.map(remove_columns=["writer_id", "hsf_id"])

    # train_torch = train.map(
    #   lambda img: {"image": transforms(img)}, input_columns="image",
    #   remove_columns=["writer_id", "hsf_id"]
    # ).with_format("torch")

    training_set = TensorDataset(data=train['image'], targets=train['character'],transform=transforms)


    # Now, you can check if you didn't make any mistakes by calling partition_torch[0]
    train_dataloader = DataLoader(training_set, batch_size=20)
    print(train_dataloader.dataset)
    train_iterator = iter(train_dataloader)

    a, b = next(train_iterator)
    print(a, b)


