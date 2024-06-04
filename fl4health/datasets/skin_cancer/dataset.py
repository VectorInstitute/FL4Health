import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

from fl4health.utils.dataset import BaseDataset


class SkinCancerDataset(BaseDataset):
    def __init__(self, data, transform=None):
        super().__init__()
        self.data = []
        self.targets = []

        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.ToTensor()

        # Load and transform all images upfront
        for item in data:
            image_path = item["img_path"]
            image = Image.open(image_path).convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
            self.data.append(image)

            target = torch.tensor(item["extended_labels"].index(1.0))
            self.targets.append(target)

        self.data = torch.stack(self.data)  # Convert list of images to a tensor
        self.targets = torch.tensor(self.targets)  # Convert list of targets to a tensor

    def __getitem__(self, item):
        data = self.data[item]
        target = self.targets[item]
        return data, target
