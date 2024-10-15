import pytest

import torch
from fl4health.losses.deep_mmd_loss import DeepMmdLoss

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

X = torch.Tensor(
    [
        [1, 1, 1],
        [3, 4, 4],
        [4, 2, 1],
        [2, 1, 4],
        [1, 2, 1],
        [3, 4, 4],
        [4, 3, 3],
        [3, 3, 2],
        [4, 4, 4],
        [4, 2, 1],
        [1, 1, 1],
    ]
).to(DEVICE)
Y = torch.Tensor(
    [
        [4, 3, 4],
        [1, 2, 2],
        [3, 4, 1],
        [1, 4, 2],
        [4, 2, 4],
        [4, 1, 2],
        [2, 2, 1],
        [2, 3, 4],
        [3, 2, 1],
        [4, 1, 4],
        [2, 2, 2],
    ]
).to(DEVICE)


def test_forward() -> None:
    torch.manual_seed(42)
    deep_mmd_loss_1 = DeepMmdLoss(device=DEVICE, input_size=3, training=True, optimization_steps=1)
    outputs_train_1 = []
    outputs_val_1 = []
    for i in range(5):
        deep_mmd_loss_1.training = True
        output = deep_mmd_loss_1(X, Y)
        outputs_train_1.append(output)
        deep_mmd_loss_1.training = False
        output = deep_mmd_loss_1(X, Y)
        outputs_val_1.append(output)

    print(outputs_train_1[0].item())

    # The output of the DeepMmdLoss in training mode should be different for each optimization step
    # as values are updated in each step
    assert pytest.approx(outputs_train_1[0].item(), abs=0.001) == 0.0584
    assert pytest.approx(outputs_train_1[1].item(), abs=0.001) == 0.0682
    assert pytest.approx(outputs_train_1[2].item(), abs=0.001) == 0.0773
    assert pytest.approx(outputs_train_1[3].item(), abs=0.001) == 0.0850
    assert pytest.approx(outputs_train_1[4].item(), abs=0.001) == 0.0914

    for i in range(len(outputs_val_1)):
        # The output of the DeepMmdLoss in evaluation mode should be the same as the output of the DeepMmdLoss in
        # training mode for the same input
        assert outputs_val_1[i] == outputs_train_1[i]

    # Reset the seed for the second DeepMmdLoss
    torch.manual_seed(42)
    deep_mmd_loss_2 = DeepMmdLoss(device=DEVICE, input_size=3, training=True, optimization_steps=5)
    output = deep_mmd_loss_2(X, Y)
    # The output of applying optimization_steps=5 should be the same as the output of applying optimization_steps=1
    # for five times for the same input
    assert pytest.approx(output.item(), abs=0.00001) == outputs_train_1[4].item() 
