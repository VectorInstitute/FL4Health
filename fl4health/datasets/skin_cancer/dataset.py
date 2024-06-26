from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import torch
import torchvision.transforms as transforms
from PIL import Image

from fl4health.utils.dataset import BaseDataset


class SkinCancerDataset(BaseDataset):
    def __init__(
        self, data: List[Dict[str, Any]], transform: Optional[transforms.Compose] = None, num_workers: int = 8
    ) -> None:
        super().__init__()
        self.data: torch.Tensor = torch.tensor([])  # Initialize as empty tensor
        self.targets: torch.Tensor = torch.tensor([])  # Initialize as empty tensor
        self.num_workers = num_workers
        self.transform = transform if transform is not None else transforms.ToTensor()
        self._load_data(data)

    def _load_image(self, item: Dict[str, Any]) -> Tuple[torch.Tensor, int]:
        image_path = item["img_path"]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        else:
            # Convert PIL Image to Tensor if no transform is provided
            image = transforms.ToTensor()(image)
        target = int(torch.tensor(item["extended_labels"]).argmax().item())  # Convert to Python int
        return image, target

    def _load_data(self, data: List[Dict[str, Any]]) -> None:
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(self._load_image, data))

        data_list, targets_list = zip(*results)
        self.data = torch.stack(list(data_list))  # Convert list of images to a tensor
        self.targets = torch.tensor(list(targets_list))  # Convert list of targets to a tensor

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.data[item]
        target = self.targets[item]
        return data, target

    def __len__(self) -> int:
        return len(self.data)
