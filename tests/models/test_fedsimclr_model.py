from pathlib import Path

import torch

from fl4health.model_bases.fedsimclr_base import FedSimClrModel
from tests.test_utils.models_for_test import LinearPredictionHead, LinearTransform


def test_pretrain_fedsimclr_model() -> None:
    model = FedSimClrModel(LinearTransform(), prediction_head=LinearPredictionHead(), pretrain=True)
    input = torch.randn(10, 2)
    output = model(input)
    # Output should just be the flattened tensor output from the feature cnn
    assert output.shape == (10, 3)


def test_finetune_fedsimclr_model() -> None:
    model = FedSimClrModel(LinearTransform(), prediction_head=LinearPredictionHead(), pretrain=False)
    input = torch.randn(10, 2)
    output = model(input)
    # output should run through the head cnn
    assert output.shape == (10, 2)


def test_load_pretrain_fedsimclr_model(tmp_path: Path) -> None:
    save_path = tmp_path.joinpath("temp_checkpoint.pkl")
    model = FedSimClrModel(LinearTransform(), prediction_head=LinearPredictionHead(), pretrain=True)
    torch.save(model, save_path)

    model = FedSimClrModel.load_pretrained_model(save_path)
    assert not model.pretrain
    assert model.encoder is not None
