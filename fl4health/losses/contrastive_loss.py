import torch


class ContrastiveLoss(torch.nn.Module):
    def __init__(
        self,
        device: torch.device,
        temperature: float = 0.5,
    ) -> None:
        """
        This constrastive loss is implemented based on https://github.com/QinbinLi/MOON.
        Contrastive loss aims to enhance the similarity between the features and their positive pairs
        while reducing the similarity between the features and their negative pairs.

        Args:
            device (torch.device): device to use for computation
            temperature (float): temperature to scale the logits
        """

        super().__init__()
        self.device = device
        self.temperature = temperature
        self.cos_sim = torch.nn.CosineSimilarity(dim=-1).to(self.device)
        self.ce_criterion = torch.nn.CrossEntropyLoss().to(self.device)

    def forward(
        self, features: torch.Tensor, positive_pairs: torch.Tensor, negative_pairs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the contrastive loss based on the features, positive pair and negative pairs. While every feature
        has a positive pair, it can have multiple negative pairs. The loss is computed based on the similarity
        between the feature and its positive pair.

        Args:
            features (torch.Tensor): Main features, shape (n_samples, n_features)
            positive_pairs (torch.Tensor): Positive pair of main features, shape (1, n_samples, n_features)
            negative_pairs (torch.Tensor): Negative pairs of main features, shape (n_pairs ,n_samples, n_features)

        Returns:
            torch.Tensor: Contrastive loss value
        """
        assert self.temperature is not None
        if len(positive_pairs) != 1:
            AssertionError(
                """Each feature can have one positive pair.
                Thus positive pairs should be a tensor of shape (1, n_samples, n_features)"""
            )
        positive_pair = positive_pairs[0]
        assert len(features) == len(positive_pair)
        posi = self.cos_sim(features, positive_pair)
        logits = posi.reshape(-1, 1)
        for negative_pair in negative_pairs:
            assert len(features) == len(negative_pair)
            nega = self.cos_sim(features, negative_pair)
            logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)
        logits /= self.temperature
        labels = torch.zeros(features.size(0)).to(self.device).long()

        return self.ce_criterion(logits, labels)
