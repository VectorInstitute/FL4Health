import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module):
    def __init__(
        self,
        device: torch.device,
        temperature: float = 0.5,
    ) -> None:
        """
        This contrastive loss is implemented based on https://github.com/QinbinLi/MOON.
        Contrastive loss aims to enhance the similarity between the features and their positive pairs
        while reducing the similarity between the features and their negative pairs.
        Args:
            device (torch.device): device to use for computation
            temperature (float): temperature to scale the logits
        """

        super().__init__()
        self.device = device
        self.temperature = temperature
        self.cosine_similarity_function = torch.nn.CosineSimilarity(dim=-1).to(self.device)
        self.cross_entropy_function = torch.nn.CrossEntropyLoss().to(self.device)

    def compute_negative_similarities(self, features: torch.Tensor, negative_pairs: torch.Tensor) -> torch.Tensor:
        """
        This function computes the cosine similarities of the batch of features provided with the set of batches of
        negative pairs.

        Args:
            features (torch.Tensor): Main features, shape (batch_size, n_features)
            negative_pairs (torch.Tensor): Negative pairs of main features, shape (n_pairs, batch_size, n_features)

        Returns:
            torch.Tensor: Cosine similarities of the batch of features provided with the set of batches of
                negative pairs. The shape is n_pairs x batch_size
        """
        # Check that features and each of the negatives pairs have the same shape
        assert features.shape == negative_pairs.shape[1:]
        # Repeat the feature tensor to compute the similarity of the feature tensor with all negative pairs.
        repeated_features = features.unsqueeze(0).repeat(len(negative_pairs), 1, 1)
        return self.cosine_similarity_function(repeated_features, negative_pairs)

    def forward(
        self, features: torch.Tensor, positive_pairs: torch.Tensor, negative_pairs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the contrastive loss based on the features, positive pair and negative pairs. While every feature
        has a positive pair, it can have multiple negative pairs. The loss is computed based on the similarity
        between the feature and its positive pair relative to negative pairs.

        Args:
            features (torch.Tensor): Main features, shape (batch_size, n_features)
            positive_pairs (torch.Tensor): Positive pair of main features, shape (1, batch_size, n_features)
            negative_pairs (torch.Tensor): Negative pairs of main features, shape (n_pairs, batch_size, n_features)

        Returns:
            torch.Tensor: Contrastive loss value
        """
        # TODO: We can extend it to support multiple positive pairs using multi-label classification

        features = features.to(self.device)
        positive_pairs = positive_pairs.to(self.device)
        negative_pairs = negative_pairs.to(self.device)

        if len(positive_pairs) != 1:
            raise AssertionError(
                "Each feature can have only one positive pair. ",
                "Thus positive pairs should be a tensor of shape (1, batch_size, n_features) ",
                f"rather than {positive_pairs.shape}",
            )

        positive_pair = positive_pairs[0]
        assert len(features) == len(positive_pair)
        # Compute similarity of the batch of features with the provided batch of positive pair features
        positive_similarity = self.cosine_similarity_function(features, positive_pair)
        # Store similarities with shape batch_size x 1
        logits = positive_similarity.reshape(-1, 1)

        # Compute the similarity of the batch of features with the collection of batches of negative pair features
        # Shape of tensor coming out is n_pairs x batch_size
        negative_pair_similarities = self.compute_negative_similarities(features, negative_pairs)
        logits = torch.cat((logits, negative_pair_similarities.T), dim=1)
        logits /= self.temperature
        labels = torch.zeros(features.size(0)).to(self.device).long()

        return self.cross_entropy_function(logits, labels)
