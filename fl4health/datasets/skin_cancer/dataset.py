import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor
from fl4health.utils.dataset import BaseDataset

class SkinCancerDataset(BaseDataset):
    def __init__(self, data, transform=None, num_workers=8) -> None:
        super().__init__()
        self.data = []
        self.targets = []

        self.num_workers = num_workers

        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.ToTensor()

        self._load_data(data)

    def _load_image(self, item) -> None:
        image_path = item["img_path"]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        target = torch.tensor(item["extended_labels"].index(1.0))
        return image, target

    def _load_data(self, data) -> None:
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(self._load_image, data))

        self.data, self.targets = zip(*results)
        self.data = torch.stack(self.data)  # Convert list of images to a tensor
        self.targets = torch.tensor(self.targets)  # Convert list of targets to a tensor

    def __getitem__(self, item) -> None:
        data = self.data[item]
        target = self.targets[item]
        return data, target

    def __len__(self) -> None:
        return len(self.data)

