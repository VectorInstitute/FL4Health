from fl4health.model_bases.sequential_split_models import SequentiallySplitExchangeBaseModel


class FedRepModel(SequentiallySplitExchangeBaseModel):
    """
    Implementation of the FedRep model structure: https://arxiv.org/pdf/2102.07078.pdf
    The architecture is fairly straightforward. The global module represents the first set of layers. These are
    learned with FedAvg. The local_prediction_head are the last layers, these are not exchanged with the server.
    The approach resembles FENDA, but vertical rather than parallel models. It also resembles MOON, but with
    partial weight exchange for weight aggregation.
    """

    def freeze_base_module(self) -> None:
        for parameters in self.base_module.parameters():
            parameters.requires_grad = False

    def unfreeze_base_module(self) -> None:
        for parameters in self.base_module.parameters():
            parameters.requires_grad = True

    def freeze_head_module(self) -> None:
        for parameters in self.head_module.parameters():
            parameters.requires_grad = False

    def unfreeze_head_module(self) -> None:
        for parameters in self.head_module.parameters():
            parameters.requires_grad = True
